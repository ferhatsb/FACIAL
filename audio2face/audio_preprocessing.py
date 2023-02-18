import glob
import os
from _utils.audio_handler import AudioHandler
from scipy.io import wavfile
import pickle
import argparse


def cross_check_existence(audio_fname_list, mesh_fname_list):
    _audio_list = [i.split('/')[-1].replace('.wav', '') for i in audio_fname_list]
    _mesh_list = [i.split('/')[-1] for i in mesh_fname_list]

    miss_sent = set(_audio_list).symmetric_difference(set(_mesh_list))

    for sent in miss_sent:
        audio_fname_list = [i for i in audio_fname_list if sent not in i]
        mesh_fname_list = [i for i in mesh_fname_list if sent not in i]

    assert len(audio_fname_list) + len(miss_sent) == 40 and len(mesh_fname_list) + len(miss_sent) == 40
    return audio_fname_list, mesh_fname_list


def process_audio(ds_path, audio, fps):
    config = {}
    config['deepspeech_graph_fname'] = ds_path
    config['audio_feature_type'] = 'deepspeech'
    config['num_audio_features'] = 29

    config['audio_window_size'] = 16
    config['audio_window_stride'] = 1
    if fps == 25:
        config['audio_window_stride'] = 2

    audio_handler = AudioHandler(config)
    processed_audios = audio_handler.process(audio, fps)
    return processed_audios


def main():
    parser = argparse.ArgumentParser(description='audio preprocessing')
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--dataset_path', type=str, default='../examples/')
    parser.add_argument('--output_path', type=str, default='../examples/audio_preprocessed')
    opt = parser.parse_args()

    fps = 30  # frame rate, same, do not need change
    dataset_path = opt.dataset_path  # The root of my audios, inside is cliton, obama....
    subjects = ['audio']  # names
    # audio_list = glob.glob(os.path.join(dataset_path, '*/audio/*.wav'))
    ds_fname = 'ds_graph/output_graph.pb'  # deep speech model

    audio4deepspeech = {}

    if opt.input_file is not None:
        audio_file = opt.input_file
        sentence = audio_file.split('/')[-1][0:-4]  # get wav name
        sample_rate, audio = wavfile.read(audio_file)  # read wav file
        tmp_audio = {sentence: {'audio': audio, 'sample_rate': sample_rate}}
        audio4deepspeech[subjects[0]] = tmp_audio
        processed_audio = process_audio(ds_fname, audio4deepspeech, fps)  # generate audio feature
    else:
        # print 'Loading audio for preprocessing...'
        for subject in subjects:
            audio_list = glob.glob(os.path.join(dataset_path, subject + '/*.wav'))  # subject file location
            tmp_audio = {}
            for audio_fname in audio_list:
                sentence = audio_fname.split('/')[-1][0:-4]  # get wav name

                sample_rate, audio = wavfile.read(audio_fname)  # read wav file
                tmp_audio[sentence] = {'audio': audio, 'sample_rate': sample_rate}
            audio4deepspeech[subject] = tmp_audio  # save format names(obama), video(0-ZCDAUSH)

        processed_audio = process_audio(ds_fname, audio4deepspeech, fps)  # generate audio feature

    for subject in processed_audio.keys():  # save audio file
        out_path = opt.output_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for sentence in processed_audio[subject]:
            out_file = os.path.join(out_path, sentence + '.pkl')
            _audio = processed_audio[subject][sentence]['audio']
            pickle.dump(_audio, open(out_file, 'wb'))


if __name__ == "__main__":
    main()
