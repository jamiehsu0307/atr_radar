from glob import glob
import main_single_image
image_list = glob('/data/yolov9/tai/*.png')
for image_path in image_list:
    main_single_image.IMAGE_PATH = image_path
    main_single_image.run('ship')