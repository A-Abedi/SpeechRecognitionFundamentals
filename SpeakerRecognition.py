from SpeechHelper import SpeechHelper
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from joblib import dump, load


def extract_mfcc(dataset_type: str = "test"):
    print("Extract MFCC {} set".format(dataset_type))

    path = "Dataset/" + dataset_type
    for speaker in os.listdir(path):
        wave_path = "{}/{}/".format(path, speaker)

        print("\t", speaker)
        for wave_file in os.listdir(wave_path):
            print("\t\t", wave_file)

            speech = SpeechHelper(wave_path + "/" + wave_file)
            MFCC = speech.mfcc()
            np.save("Outputs/mfcc/{}/{}/{}.npy".format(dataset_type, speaker, wave_file.split(".")[0]), MFCC)


def train():
    print("Train")

    path = "Outputs/mfcc/train"

    model_holder = dict()
    for speaker in os.listdir(path):
        wave_path = "{}/{}/".format(path, speaker)

        print("\t", speaker)

        MFCC_vectors = list()
        for wave_file in os.listdir(wave_path):
            wave_file_name = wave_file.split(".")[0]
            print("\t\t", wave_file_name)

            MFCC = np.load("{}/{}/{}.npy".format(path, speaker, wave_file_name))
            MFCC_vectors += MFCC.tolist()

        gm = GaussianMixture(n_components=32, random_state=1212, covariance_type="diag").fit(MFCC_vectors)
        model_holder[speaker] = gm

        dump(gm, "Outputs/model/{}.joblib".format(speaker))

    return model_holder


def test():
    print("Test")

    path = "Outputs/mfcc/test"
    result = {}

    models = {}
    path_model = "Outputs/model"
    for model in os.listdir(path_model):
        gm = load("{}/{}".format(path_model, model))
        models[model.split(".")[0]] = gm

    for speaker in os.listdir(path):
        wave_path = "{}/{}/".format(path, speaker)
        print("\t", speaker)

        result[speaker] = {}
        for wave_file in os.listdir(wave_path):
            wave_file_name = wave_file.split(".")[0]
            print("\t\t", wave_file_name)

            MFCC = np.load("{}/{}/{}.npy".format(path, speaker, wave_file_name))

            models_result = {}
            for speaker_ind, model in models.items():
                score = model.score(MFCC)
                models_result[speaker_ind] = score

            result[speaker][wave_file_name] = models_result
    return result
