CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

IMAGE_TOKEN_INDEX = -200
BOX_TOKEN_INDEX = -500
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# ======================================================================================================
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<im_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
VIDEO_PLACEHOLDER = "<video-placeholder>"
# ======================================================================================================
DEFAULT_BBOX_TOKEN = "<bbox>"
# =======================================================================================================
MAX_IMAGE_LENGTH = 16
MAX_VIDEO_LENGTH = 1  # current video datasets only have 1 video?

PAD_LENGTH = 620

IN_THE_VIDEO_QUESTION = [
    'How would you characterize the <bbox>in this video?',
    'What action is the <bbox>performing in this video?',
    'Throughout this video, what is the behavior of this object <bbox>?',
    'How is the <bbox>acting in this video?',
    'What is the motion of the object in this zone <bbox>in this video?',
    'What is the <bbox>doing in this video?',
    'As this video plays, what is this object <bbox>doing?',
    'In what way is the <bbox>behaving in this video?',
    'While this video is running, what is this object <bbox>doing?',
    'What are the characteristics of the <bbox>in this video?',
    'How does the <bbox>move in this video?',
    'During this video, what is this object <bbox>accomplishing?',
    'What is the activity of the object in this area <bbox>in this video?',
    'What is this object <bbox>doing throughout this video?',
    'What is the <bbox>doing in this video?',
    'How is the object in this part <bbox>behaving in this video?',
    'What is the behavior of this object <bbox>in this video?',
    'How does the object in this section <bbox>act in this video?',
    'What is the action of this object <bbox>in this video?',
    'How is the object in this segment <bbox>moving in this video?',
    'What is the movement of this object <bbox>in this video?',
    'What is the <bbox>engaged in in this video?',
    'How does the object in this zone <bbox>operate in this video?',
    'What is the operation of this object <bbox>in this video?'
]


TRACKING_LIST_NOTION=[
    "This is the region's video tracking list:",
    "There is the region's video tracking list:",
    "You are provided the region's video tracking list:",
]
