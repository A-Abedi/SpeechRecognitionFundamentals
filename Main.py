import SpeakerRecognition

SpeakerRecognition.extract_mfcc("train")
SpeakerRecognition.extract_mfcc("test")

models = SpeakerRecognition.train()
results = SpeakerRecognition.test()

top1 = 0
top3 = 0

print("Results: ")
for speaker, result in results.items():
    print(speaker)

    for file_name, models_result in result.items():
        print("\t", "{}".format(file_name))
        models_result_sorted = sorted(models_result.items(), key=lambda x: x[1], reverse=True)[:3]

        top1 += 1 if models_result_sorted[0][0] == speaker else 0

        for model_name, score in models_result_sorted:
            top3 += 1 if model_name == speaker else 0
            print("\t\t", "{} : {}".format(model_name, round(score, 3)))

print("\ntop1:", top1, "/60")
print("top3:", top3, "/60")
