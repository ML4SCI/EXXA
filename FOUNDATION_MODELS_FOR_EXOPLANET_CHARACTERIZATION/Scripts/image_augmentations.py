import albumentations as A
from albumentations.pytorch import ToTensorV2

transform_mae = A.Compose(
    [
        A.Normalize(
            mean=0, 
            std=1, 
            max_pixel_value=1.0,  # Adjust based on your image values
            always_apply=True,
        ),
        A.LongestMaxSize(max_size=256, always_apply=True),
        A.PadIfNeeded(
            min_height=256, 
            min_width=256, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, 
            always_apply=True
        ),
        A.Resize(
            height=224,
            width=224,
            always_apply=True
        ),
        ToTensorV2(),
    ]
)