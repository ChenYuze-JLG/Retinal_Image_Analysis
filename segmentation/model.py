import segmentation_models_pytorch as smp

def build_model(encoder='resnet18', encoder_weights='imagenet'):
    """
    构建 U-Net 模型。

    参数:
    - encoder (str): 编码器的名称 (e.g., 'resnet18', 'resnet34').
    - encoder_weights (str): 预训练权重的来源 (e.g., 'imagenet').

    返回:
    - model: PyTorch 模型实例。
    """
    model = smp.Unet(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        in_channels=3,  # 输入图像为RGB三通道
        classes=1,      # 输出为单通道的分割掩码 (血管 vs. 背景)
        activation='sigmoid' # 输出激活函数，将值映射到 [0, 1]
    )
    return model

if __name__ == '__main__':
    # 测试模型构建
    model = build_model()
    print("模型结构:")
    print(model)
    print("\n模型构建成功！")
