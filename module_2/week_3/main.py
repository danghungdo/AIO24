import numpy as np


def create_train_data():
    data = [['Sunny', 'Hot', 'High', 'Weak', 'No'],
            ['Sunny', 'Hot', 'High', 'Strong', 'No'],
            ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
            ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
            ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
            ['Overcast', 'Mild', 'High', 'Weak', 'No'],
            ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'Normal', 'Weak', 'Yes']
            ]
    return np.asarray(data)


def compute_prior_probability(train_data):
    y_unique = ['Yes', 'No']
    prior_probability = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_probability[i] = np.sum(
            train_data[:, -1] == y_unique[i]) / len(train_data)
    return prior_probability


def compute_conditional_probability(train_data):
    y_unique = ['Yes', 'No']
    conditional_probability = []
    list_x_name = []
    for i in range(train_data.shape[1]-1):
        list_x_name.append(np.unique(train_data[:, i]))
        x_conditional_probability = {}
        for y in y_unique:
            y_data = train_data[train_data[:, -1] == y]
            probs = []
            for x in list_x_name[i]:
                probs.append(np.sum(y_data[:, i] == x) / len(y_data))
            x_conditional_probability[y] = probs
        conditional_probability.append(x_conditional_probability)
    return conditional_probability, list_x_name


def get_index_from_value(feature_name, list_features):
    return np.nonzero(list_features == feature_name)[0][0]


def train_naive_bayes(train_data):
    prior_probability = compute_prior_probability(train_data)
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)
    return prior_probability, conditional_probability, list_x_name


def predict_play_tennis(x, list_x_name, prior_probability, conditional_probability):
    x1 = get_index_from_value(x[0], list_x_name[0])
    x2 = get_index_from_value(x[1], list_x_name[1])
    x3 = get_index_from_value(x[2], list_x_name[2])
    x4 = get_index_from_value(x[3], list_x_name[3])
    p_yes = prior_probability[0]
    p_no = prior_probability[1]

    p_yes *= conditional_probability[0]["Yes"][x1]
    p_yes *= conditional_probability[1]["Yes"][x2]
    p_yes *= conditional_probability[2]["Yes"][x3]
    p_yes *= conditional_probability[3]["Yes"][x4]

    p_no *= conditional_probability[0]["No"][x1]
    p_no *= conditional_probability[1]["No"][x2]
    p_no *= conditional_probability[2]["No"][x3]
    p_no *= conditional_probability[3]["No"][x4]
    if p_yes > p_no:
        return 1
    else:
        return 0


if __name__ == '__main__':
    train_data = create_train_data()
    print(train_data)
    # Question 14
    prior_probability = compute_prior_probability(train_data)
    print("P(play tennis = Yes):", prior_probability[0])
    print("P(play tennis = No):", prior_probability[1])
    # Question 15
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)
    print("x1 = ", list_x_name[0])
    print("x2 = ", list_x_name[1])
    print("x3 = ", list_x_name[2])
    print("x4 = ", list_x_name[3])
    print(conditional_probability)
    # Question 16
    outlook = list_x_name[0]
    i1 = get_index_from_value('Overcast', outlook)
    i2 = get_index_from_value('Rain', outlook)
    i3 = get_index_from_value('Sunny', outlook)
    print(i1, i2, i3)
    # Question 17
    x1 = get_index_from_value("Sunny", list_x_name[0])
    print("P(Outlook = Sunny | Play Tennis = Yes):", np.round(
        conditional_probability[0]["Yes"][x1], 2))
    # Question 18
    print("P(Outlook = Sunny | Play Tennis = No):", np.round(
        conditional_probability[0]["No"][x1], 2))
    # Question 19
    X = ['Sunny', 'Cool', 'High', 'Strong']
    prior_probability, conditional_probability, list_x_name = train_naive_bayes(
        train_data)
    pred = predict_play_tennis(
        X, list_x_name, prior_probability, conditional_probability)
    if pred:
        print("Ad should go!")
    else:
        print("Ad should not go!")
