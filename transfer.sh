#!/bin/sh

FACIAL_FOLDER=~/Development/FACIAL
DATA_FOLDER=~/Development/facial_data
VIDEO_PREPROCESS_FOLDER=$DATA_FOLDER/video_preprocess
AUDIO_PREPROCESS_FOLDER=$DATA_FOLDER/audio_preprocessed

audio_preprocess(){
  echo "audio preprocess..."
  cd $FACIAL_FOLDER/audio2face
  python audio_preprocessing.py --input_file $VIDEO_PREPROCESS_FOLDER/test1.wav --output_path $AUDIO_PREPROCESS_FOLDER
}

test_audio_to_face(){
  echo "test audio to face"
  cd $FACIAL_FOLDER/audio2face
  python test.py --audiopath $AUDIO_PREPROCESS_FOLDER/test1.pkl \
    --checkpath $DATA_FOLDER/audio2face/checkpoint/train1/Gen-10.mdl \
    --outpath $DATA_FOLDER/test_result/
}

render_3D_face(){
  echo "render 3D face"
  cd $FACIAL_FOLDER/face_render/
  python rendering_gaosi.py --train_params_path $VIDEO_PREPROCESS_FOLDER/train1_posenew.npz \
    --net_params_path $DATA_FOLDER/test_result/test1.npz \
    --outpath $DATA_FOLDER/rendering/
}

test_face_to_video(){
  cd $FACIAL_FOLDER/face2vid

  mkdir -p $DATA_FOLDER/face2vid/datasets/train3
  rm -rf $DATA_FOLDER/face2vid/datasets/train3/test_A $DATA_FOLDER/face2vid/datasets/train3/test_B
  cp -r $DATA_FOLDER/rendering/test1 $DATA_FOLDER/face2vid/datasets/train3/test_A
  cp -r $DATA_FOLDER/rendering/test1 $DATA_FOLDER/face2vid/datasets/train3/test_B

  python test_video.py --test_id_name test1 --blink_path $DATA_FOLDER/test_result/test1.npz \
    --name train3 --model pose2vid --dataroot $DATA_FOLDER/face2vid/datasets/train3/ \
    --results_dir $DATA_FOLDER/results \
    --checkpoints_dir $DATA_FOLDER/checkpoints \
    --which_epoch latest --netG local --ngf 32 --label_nc 0 --n_local_enhancers 1 \
    --no_instance --resize_or_crop resize
}

create_video(){
  audio_path=$VIDEO_PREPROCESS_FOLDER/test1.wav
  video_new=$DATA_FOLDER/results/test1/test_1.avi
  output=$DATA_FOLDER/results/test1/test_1_audio.avi
  output_mp4=$DATA_FOLDER/results/test1/test_1_audio.mp4
  ffmpeg -i $video_new -i $audio_path -c copy $output
  ffmpeg -i $output  $output_mp4
}

audio_preprocess
test_audio_to_face
render_3D_face
test_face_to_video
create_video