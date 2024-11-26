import torch
import math


def encode_frames(input_tensor, vae):
    """
    使用 VAE 编码每个帧。

    参数:
        input_tensor (torch.Tensor): 输入 tensor，形状为 [batch_size, channels, frames, height, width]
        vae (AutoEncoderKL): 预训练的 VAE 模型

    返回:
        torch.Tensor: 编码后的 latent images，形状为 [batch_size, frames, latent_channels, latent_height, latent_width]
    """
    batch_size, channels, frames, height, width = input_tensor.shape

    if channels == 6:
        # 如果 channels 是 6，将其拆分为两张图片分别编码
        image1 = input_tensor[:, :3, :, :, :]  # 第一张图片
        image2 = input_tensor[:, 3:, :, :, :]  # 第二张图片

        # 将两张图片分别编码
        encoded_image1 = encode_single_frame(image1, vae)
        encoded_image2 = encode_single_frame(image2, vae)

        # 将编码后的结果拼接
        latent_images = torch.cat((encoded_image1, encoded_image2), dim=2)  # [batch_size, frames*2, latent_channels, latent_height, latent_width]
    else:
        # 如果 channels 是 3，直接编码
        latent_images = encode_single_frame(input_tensor, vae)

    return latent_images

def encode_single_frame(image_tensor, vae):
    """
    使用 VAE 编码单个帧的函数。

    参数:
        image_tensor (torch.Tensor): 输入 tensor，形状为 [batch_size, channels, frames, height, width]
        vae (AutoEncoderKL): 预训练的 VAE 模型

    返回:
        torch.Tensor: 编码后的 latent images，形状为 [batch_size, frames, latent_channels, latent_height, latent_width]
    """
    batch_size, channels, frames, height, width = image_tensor.shape

    # 将输入 tensor 变形为 [batch_size * frames, channels, height, width]
    image_tensor_reshaped = image_tensor.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

    # VAE 编码
    with torch.no_grad():
        # Normalize to [-1, 1]
        image_tensor_reshaped = image_tensor_reshaped.mul_(2).sub_(1)
        latents = vae.encode(image_tensor_reshaped).latent_dist.sample().mul_(0.18215)

    # 将编码后的 latents 变形为 [batch_size, frames, latent_channels, latent_height, latent_width]
    latent_images = latents.view(batch_size, frames, 4, 32, 32)
    return latent_images

def stitch_frames(latent_images):
    """
    将帧拼接成正方形的图。

    参数:
        latent_images (torch.Tensor): 编码后的 latent images，形状为 [batch_size, frames, latent_channels, latent_height, latent_width]

    返回:
        torch.Tensor: 拼接后的图像，形状为 [batch_size, latent_channels, num_rows * latent_height, num_cols * latent_width]
    """
    batch_size, frames, latent_channels, latent_height, latent_width = latent_images.shape

    # 确保 frames 是一个平方数
    num_rows = num_cols = int(math.sqrt(frames))

    # 重新排列 tensor，以便直接拼接
    latent_images = latent_images.permute(0, 2, 1, 3, 4)  # [batch_size, latent_channels, frames, latent_height, latent_width]

    # 将帧排列成正方形
    latent_images = latent_images.reshape(batch_size, latent_channels, num_rows, num_cols, latent_height, latent_width)
    latent_images = latent_images.permute(0, 1, 2, 4, 3, 5)  # [batch_size, latent_channels, num_rows, latent_height, num_cols, latent_width]

    final_images = latent_images.reshape(batch_size, latent_channels, num_rows * latent_height, num_cols * latent_width)
    return final_images

def reshape_audio_sequences_for_dit(tensor): # [10, 4, 1, 50, 384] -> [10, 200, 384]
    batch_size, T, channels, chunks, feature_size = tensor.shape
    reshaped_tensor = tensor.view(batch_size, T * channels * chunks, feature_size)
    return reshaped_tensor

def unstitch_frames(stitched_images, T):
    """
    将拼接的图像拆分回原来的帧。

    参数:
        stitched_images (torch.Tensor): 拼接后的图像，形状为 [batch_size, latent_channels, num_rows * latent_height, num_cols * latent_width]
        frames (int): 原来帧的数量，必须是一个平方数

    返回:
        torch.Tensor: 拆分后的帧，形状为 [batch_size, frames, latent_channels, latent_height, latent_width]
    """
    batch_size, latent_channels, stitched_height, stitched_width = stitched_images.shape

    # 确保 frames 是一个平方数
    num_rows = num_cols = int(math.sqrt(T))

    latent_height = stitched_height // num_rows
    latent_width = stitched_width // num_cols

    # 重新排列 tensor，以便拆分
    stitched_images = stitched_images.reshape(batch_size, latent_channels, num_rows, latent_height, num_cols, latent_width)
    stitched_images = stitched_images.permute(0, 1, 2, 4, 3, 5)  # [batch_size, latent_channels, num_rows, num_cols, latent_height, latent_width]

    # 将正方形拆分回帧
    latent_images = stitched_images.reshape(batch_size, latent_channels, T, latent_height, latent_width)
    latent_images = latent_images.permute(0, 2, 1, 3, 4)  # [batch_size, T, latent_channels, latent_height, latent_width]

    return latent_images

def decode_frames(input_tensor, vae):
    batch_size, T, latent_channels, height, width = input_tensor.shape
    input_tensor = input_tensor.reshape(batch_size * T, latent_channels, height, width)
    with torch.no_grad():
        images = vae.decode(input_tensor / 0.18215).sample
        
    batch_size_times_T, image_channels, image_height, image_width = images.shape
    return images.view(batch_size, T, image_channels, image_height, image_width)