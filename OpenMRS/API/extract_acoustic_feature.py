import librosa

def extract_acoustic_feature(input_file,
                             num_mfcc = 128,
                             number_mels = 128,
                             frequency_max = 8000):
    """ Load the music file
        Extract Mel Frequency Cepstrum Coefficient
        :return feature: a matrix contains extracted mfcc
        :Author: Chris Hu
    """
    #load music from librosa example
    try:
        y,sr = librosa.load(input_file)
    except:
        print("File loading error")
        return 0
    
    #feature extraction(mfcc)
    #librosa.feature.mfcc(y=y, sr=sr)
    
    #log-power Mel spectrogram
    S = librosa.feature.melspectrogram(y = y, sr = sr,
                                       n_mels = number_mels,
                                       fmax = frequency_max)
    #feature extraction(mfcc)
    feature = librosa.feature.mfcc(S = librosa.logamplitude(S),
                                   n_mfcc = num_mfcc)
    
    return feature
