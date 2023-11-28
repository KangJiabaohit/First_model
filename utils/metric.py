def metric(prediction, targets, labels):
    right_num = 0
    total_num = len(prediction[0])
    prediction_txt = []
    targets_txt = []
    targets = targets.to(int)
    assert len(prediction[0]) == len(targets)
    for i in range(len(prediction[0])):
        if prediction[1][i] == targets[i]:
            right_num +=1
    precision = (right_num +0.0) / total_num
    for i in range(len(prediction[0])):
        prediction_txt.append(labels[prediction[1][i]])
        targets_txt.append(labels[targets[i]])
    print("The first five predictions:")
    for i in range(5):
        print('Prediction:', prediction_txt[i], 'Targets:', targets_txt[i])
    print("precision:", precision)
    return precision