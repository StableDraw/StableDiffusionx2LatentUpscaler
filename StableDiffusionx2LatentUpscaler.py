import io
import torch
import numpy
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

def load_img(binary_data, max_dim):
    image = Image.open(io.BytesIO(binary_data)).convert("RGB")
    orig_w, orig_h = image.size
    print(f"Загружено входное изображение размера ({orig_w}, {orig_h})")
    cur_dim = orig_w * orig_h
    if cur_dim > max_dim:
        k = cur_dim / max_dim
        sk = float(k ** (0.5))
        w, h = int(orig_w / sk), int(orig_h / sk)
    else:
        w, h = orig_w, orig_h
    w, h = map(lambda x: x - x % 64, (w, h))  # изменение размера в целое число, кратное 64-м
    if w == 0 and orig_w != 0:
        w = 64
    if h == 0 and orig_h != 0:
        h = 64
    if (w, h) != (orig_w, orig_h):
        image = image.resize((w, h), resample = Image.LANCZOS)
        print(f"Размер изображения изменён на ({w}, {h} (w, h))")
    else:
        print(f"Размер исходного изображения не был изменён")
    return image

class StableDiffusionLatentUpscalePipeline(DiffusionPipeline):
    def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: EulerDiscreteScheduler,):
        super().__init__()
        self.register_modules(vae = vae, text_encoder = text_encoder, tokenizer = tokenizer, unet = unet, scheduler = scheduler)

    def enable_sequential_cpu_offload(self, gpu_id = 0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Пожалуйста, установите accelerate при помощи 'pip install accelerate'")
        device = torch.device(f"cuda:{gpu_id}")
        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model != None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device") and module._hf_hook.execution_device is not None):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_inputs = self.tokenizer(prompt, padding = "max_length", max_length = self.tokenizer.model_max_length, truncation = True, return_length = True, return_tensors = "pt")
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding = "longest", return_tensors = "pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            print(f"Следующая часть вашего ввода была усечена, потому что CLIP может обрабатывать последовательности только до {self.tokenizer.model_max_length} токенов: {removed_text}")
        text_encoder_out = self.text_encoder(text_input_ids.to(device), output_hidden_states = True)
        text_embeddings = text_encoder_out.hidden_states[-1]
        text_pooler_out = text_encoder_out.pooler_output
        # Получить безусловные эмбединги для классификации свободного управления
        if do_classifier_free_guidance:
            if negative_prompt == "":
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(f"'negative_prompt': {negative_prompt} имеет размер батча {len(negative_prompt)}, но 'описание': {prompt} размер батча {batch_size}. Пожалуйста убедитесь, что 'negative_prompt' соответствует размеру батча 'prompt'")
            else:
                uncond_tokens = negative_prompt
            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(uncond_tokens, padding = "max_length", max_length = max_length, truncation = True, return_length = True, return_tensors = "pt")
            uncond_encoder_out = self.text_encoder(uncond_input.input_ids.to(device), output_hidden_states = True)
            uncond_embeddings = uncond_encoder_out.hidden_states[-1]
            uncond_pooler_out = uncond_encoder_out.pooler_output
            # Для классификации свободного управления нужно сделать два прямых прохода. Здесь мы объединяем безусловные и текстовые встраивания в один пакет, чтобы избежать двух прямых проходов
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            text_pooler_out = torch.cat([uncond_pooler_out, text_pooler_out])
        return text_embeddings, text_pooler_out

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # Мы всегда приводим к float32, поскольку это не вызывает значительных накладных расходов и совместимо с bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height, width)
        if latents == None:
            latents = randn_tensor(shape, generator = generator, device=device, dtype = dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Нераспознанная латентная форма, получено {latents.shape}, ожидалось {shape}")
            latents = latents.to(device)
        # Масштабировать начальный шум по стандартному отклонению, требуемому планировщиком
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(self, prompt, binary_data, opt, generator = None, latents = None, output_type = "pil", return_dict = True, callback = None, callback_steps = 1):
        # 1. Определение вызываемых параметров
        batch_size = 1
        device = self._execution_device
        # Здесь "scale" определяется аналогично весу наведения "w" в уравнении (2) соответствует отсутствию свободного наведения классификатора
        guidance_scale = opt["scale"]
        do_classifier_free_guidance = guidance_scale > 1.0
        if guidance_scale == 0:
            prompt = [""] * batch_size
        # 2. Кодирование входного описания
        negative_prompt = opt["negative_prompt"]
        text_embeddings, text_pooler_out = self._encode_prompt(prompt, device, do_classifier_free_guidance, negative_prompt)
        # 3. Обработка изображения
        image = torch.from_numpy(2.0 * (numpy.array(load_img(binary_data, opt["max_dim"])).astype(numpy.float32) / 255.0)[None].transpose(0, 3, 1, 2) - 1.)
        image = image.to(dtype = text_embeddings.dtype, device = device)
        if image.shape[1] == 3:
            # Кодировать изображение, если оно еще не находится в латентном пространстве
            image = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        # 4. Установка временных шагов
        num_inference_steps = opt["steps"]
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = image[None, :] if image.ndim == 3 else image
        image = torch.cat([image] * batch_multiplier)
        # 5. Добавление шума для изображения. Этот шаг теоретически может улучшить работу модели на входных данных вне распределения, но в основном он просто заставляет ее меньше соответствовать входным данным
        noise_level = torch.tensor([opt["noise"]], dtype = torch.float32, device = device)
        noise_level = torch.cat([noise_level] * image.shape[0])
        inv_noise_level = (noise_level ** 2 + 1) ** (-0.5)
        image_cond = torch.nn.functional.interpolate(image, scale_factor = opt["outscale"], mode = "nearest") * inv_noise_level[:, None, None, None]
        image_cond = image_cond.to(text_embeddings.dtype)
        noise_level_embed = torch.cat([torch.ones(text_pooler_out.shape[0], 64, dtype = text_pooler_out.dtype, device = device), torch.zeros(text_pooler_out.shape[0], 64, dtype = text_pooler_out.dtype, device = device)], dim = 1)
        timestep_condition = torch.cat([noise_level_embed, text_pooler_out], dim = 1)
        # 6. Подготавливаются латентные переменные
        height, width = image.shape[2:]
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(batch_size, num_channels_latents, height * opt["outscale"], width * opt["outscale"], text_embeddings.dtype, device, generator, latents)
        # 7. Проверка того, что размеры изображения и латенты совпадают
        num_channels_image = image.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(f"Некорректные настройки конфигурации! Конфигурация 'pipeline.unet': {self.unet.config} ожидалось {self.unet.config.in_channels} но получено 'num_channels_latents': {num_channels_latents} + 'num_channels_image': {num_channels_image} = {num_channels_latents + num_channels_image}. Пожалуйста сверьте конфигурацию 'pipeline.unet' или входной параметр 'image'")
        # 9. Цикл шумоподавления
        num_warmup_steps = 0
        with self.progress_bar(total = num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                sigma = self.scheduler.sigmas[i]
                # Расширить латенты, если производится классификация свободного исполнения
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                scaled_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                scaled_model_input = torch.cat([scaled_model_input, image_cond], dim = 1)
                # Параметр предварительного кондиционирования, основанный на Karras полностью
                timestep = torch.log(sigma) * 0.25
                noise_pred = self.unet(scaled_model_input, timestep, encoder_hidden_states = text_embeddings, timestep_cond = timestep_condition).sample
                # В исходном репозитории выходные данные содержат неиспользуемый канал дисперсии
                noise_pred = noise_pred[:, :-1]
                # Применить предварительное кондиционирование на основе таблицы 1 в Karras полностью
                inv_sigma = 1 / (sigma ** 2 + 1)
                noise_pred = inv_sigma * latent_model_input + self.scheduler.scale_model_input(sigma, t) * noise_pred
                # Осуществлять управление
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # Вычисление предыдущей шумовой выборки x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                # Вызов обратного вызова, если он предоставлен
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback != None and i % callback_steps == 0:
                        callback(i, t, latents)
        # 10. Постобработка
        image = self.decode_latents(latents)
        # 11. Конвертация в PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        if not return_dict:
            return (image,)
        image = image[0]
        buf = io.BytesIO()
        image.save(buf, format = "PNG")
        b_data = buf.getvalue()
        image.close
        torch.cuda.empty_cache()
        return b_data

def Stable_diffusion_upscaler_xX(init_img_binary_data, caption, params):
    upscaler_pipeline = StableDiffusionLatentUpscalePipeline
    upscaler = upscaler_pipeline.from_pretrained("configs", torch_dtype = torch.float16)
    upscaler.to("cuda")
    prompt = caption
    generator = torch.manual_seed(params["seed"])
    return upscaler(prompt = prompt, binary_data = init_img_binary_data, generator = generator, opt = params)

if __name__ == '__main__':
    params = {
        "negative_prompt": "",      #Негативное описание (если без него, то "")
        "steps": 20,                #Шаги, от 2 до 250
        "seed": 33,                 #От 0 до 1000000
        "noise": 0,                 #Удаление шума (от 0 до 350)  
        "outscale": 2,              #Величина того, во сколько раз увеличть разшрешение изображения (рекомендуется 2)
        "scale": 0,                 #От 0 до 30    
        "max_dim": pow(1024, 2)     #Я не могу генерировать на своей видюхе картинки больше 512 на 512 для x4 и 512 на 512 для x2
    }

    with open("img.png", "rb") as f:
        init_img_binary_data = f.read()
    prompt = "Digital hight resolution photo of a man"
    binary_data = Stable_diffusion_upscaler_xX(init_img_binary_data, prompt, params)
    Image.open(io.BytesIO(binary_data)).save("big.png")