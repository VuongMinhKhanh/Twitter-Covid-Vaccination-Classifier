import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, \
    recall_score

pd.set_option('max_colwidth', 500)
# %matplotlib inline

# functions
def text_preprocessing(text):
    # Remove tags
    text = re.sub(r'@\s?\w+', ' ', text)
    # Remove urls
    text = re.sub(r'https?://\S+', ' ', text)
    # Remove punctuation(except ' and ’)
    text = re.sub(r'[^’\'\s\w]', ' ', text)
    # Remove underscore characters
    text = re.sub(r'_', ' ', text)
    # Remove \r and \n
    text = re.sub(r'[\r\n]', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from start
    text = re.sub(r'^[a-zA-Z]\s+', '', text)
    # Remove single characters from end
    text = re.sub(r'\s+[a-zA-Z]$', '', text)
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # Convert multiple spaces to single space and remove beginning and end spaces
    text = re.sub(r' +', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()

    return text


def remove_stopwords(text, stopwords):
    tokens = text.split(' ')
    return ' '.join([w for w in tokens if w not in stopwords])


def plot_learning_curve(X_train, y_train, X_val, y_val, model, train_sizes, title):
    plt.clf()
    train_scores = []
    validation_scores = []
    for train_size in train_sizes:
        model.fit(X_train[:train_size], y_train[:train_size])
        y_true_pred = model.predict(X_train[:train_size])
        train_scores += [f1_score(y_train[:train_size], y_true_pred, average='weighted')]
        y_pred = model.predict(X_val)
        validation_scores += [f1_score(y_val, y_pred, average='weighted')]
    # plt.grid()
    # plt.plot(train_sizes, train_scores, "o-", label="Train")
    # plt.plot(train_sizes, validation_scores, "o-", label="Test")
    # plt.xlabel("Training examples")
    # plt.ylabel("F1 Score")
    # plt.title(title)
    # plt.legend()
    # plt.show()


def evaluate_model(model, X_test, y_test):
    y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    # print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    # print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    # cm = confusion_matrix(y_test, y_pred)
    # cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    # cm_disp.plot()
    # plt.show()


# Data loading
train_dataset = pd.read_csv("data/vaccine_train_set.csv")
train_dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
# print(train_dataset)

# try removing data of label 0 and 2 to label 1's number
# Find the indices of the rows with label 0
indices_to_remove_0 = train_dataset.index[train_dataset['label'] == 0].tolist()

# Select 10 of these indices to remove (make sure you have at least 10)
indices_to_remove_0 = indices_to_remove_0[:int((len(train_dataset.index[train_dataset['label'] == 0]) - len(train_dataset.index[train_dataset['label'] == 1])) )]
# indices_to_remove_0 = indices_to_remove_0[:1000]
# Drop these rows from the DataFrame
train_dataset = train_dataset.drop(indices_to_remove_0)

indices_to_remove_2 = train_dataset.index[train_dataset['label'] == 2].tolist()
indices_to_remove_2 = indices_to_remove_2[:int((len(train_dataset.index[train_dataset['label'] == 2]) - len(train_dataset.index[train_dataset['label'] == 1])))]
# indices_to_remove_2 = indices_to_remove_2[:1000]
#
train_dataset = train_dataset.drop(indices_to_remove_2)
# print(train_dataset)

# train data shuffling
train_dataset = pd.DataFrame(train_dataset)
train_dataset = train_dataset.sample(frac=1)

test_dataset = pd.read_csv("data/vaccine_validation_or_test_set.csv")
test_dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
# print(test_dataset)

# Data visualization
trainLabelCounts = train_dataset.groupby('label').count()
trainLabelCounts.plot.bar(rot=0)
plt.show()

# Text preprocessing
train_dataset['tweet'] = train_dataset['tweet'].apply(
    lambda tweet: text_preprocessing(tweet))  # lambda tweet from ['tweet']
test_dataset['tweet'] = test_dataset['tweet'].apply(lambda tweet: text_preprocessing(tweet))

# Stopwords loading
nltk.download('stopwords')
# print(stopwords.words('english'))

# Remove stopwords
train_dataset['tweet'] = train_dataset['tweet'].apply(lambda tweet: remove_stopwords(tweet, stopwords.words("english")))
test_dataset['tweet'] = test_dataset['tweet'].apply(lambda tweet: remove_stopwords(tweet, stopwords.words("english")))
# print(train_dataset)

# Model 1: Count vectorizer and Logistic Regression
# Text vectorization
# vectorizer = CountVectorizer().fit(train_dataset['tweet'].values)
# X_train = vectorizer.transform(train_dataset['tweet'].values)
# # print(X_train)
# y_train = train_dataset['label'].values
# X_test = vectorizer.transform(test_dataset['tweet'].values)
# y_test = test_dataset['label'].values

# Training phase
# clf = LogisticRegression(max_iter=5000, tol=1e-8, multi_class='multinomial')
# plot_learning_curve(X_train, y_train, X_test, y_test, clf, np.linspace(10, X_train.shape[0], 15, dtype=np.int64))
# # print(np.linspace(10, X_train.shape[0], 15, dtype=np.int64), "Model 1")
# # [10  1150  2290  3431  4571  5712  6852  7992  9133 10273 11414 12554 13695 14835 15976]
# evaluate_model(clf, X_test, y_test)


# Model 2: Count vectorizer with min_df=4, max_df=0.3 and Logistic Regression
# vectorizer = CountVectorizer(min_df=4, max_df=0.3).fit(train_dataset['tweet'].values)
# X_train = vectorizer.transform(train_dataset['tweet'].values)
# y_train = train_dataset['label'].values
# X_test = vectorizer.transform(test_dataset['tweet'].values)
# y_test = test_dataset['label'].values
# clf = LogisticRegression(max_iter=5000, tol=1e-8, multi_class='multinomial')
# plot_learning_curve(X_train, y_train, X_test, y_test, clf, np.linspace(10, X_train.shape[0], 15, dtype=np.int64), "Model 2")
# evaluate_model(clf, X_test, y_test)


# Model 3: Tf-Idf vectorizer with min_df=4, max_df=0.3 and Logistic Regression
# vectorizer = TfidfVectorizer(min_df=4, max_df=0.3).fit(train_dataset['tweet'].values)
# X_train = vectorizer.transform(train_dataset['tweet'].values)
# y_train = train_dataset['label'].values
# X_test = vectorizer.transform(test_dataset['tweet'].values)
# y_test = test_dataset['label'].values
# clf = LogisticRegression(max_iter=5000, tol=1e-8, multi_class='multinomial')
# plot_learning_curve(X_train, y_train, X_test, y_test, clf, np.linspace(10, X_train.shape[0], 15, dtype=np.int64), "Model 3")
# evaluate_model(clf, X_test, y_test)


# Best model: Tf-Idf vectorizer with min_df=4, max_df=0.3 and Logistic Regression with C=1.83
vectorizer = TfidfVectorizer(min_df=4, max_df=0.3).fit(train_dataset['tweet'].values)
X_train = vectorizer.transform(train_dataset['tweet'].values)
y_train = train_dataset['label'].values
X_test = vectorizer.transform(test_dataset['tweet'].values)
y_test = test_dataset['label'].values
clf = LogisticRegression(max_iter=5000, tol=1e-8, C=1.83, multi_class='multinomial')
plot_learning_curve(X_train, y_train, X_test, y_test, clf, np.linspace(10, X_train.shape[0], 15, dtype=np.int64), "Best Model")
evaluate_model(clf, X_test, y_test)

# self test
tweets = np.array(["Fuck these vaccines. Lunatic thoughts from those thinking those Pfizers can cure!",
"COVID vaccines are now available for all age groups. Check your local health department for more information.",
"Reading up on the latest research regarding COVID vaccines. It's interesting to see science in progress.",
"There's a town hall meeting about COVID vaccines tonight. It's a good opportunity to hear different perspectives.",
"I'm really skeptical about these COVID vaccines. I don't think enough time has passed to understand the long-term effects.",
"Every time I see a news story about COVID vaccines, I can't help but worry about the side effects they're not telling us about.",
"My trust in the COVID vaccines is low. I feel like there's a lot of pressure to get them without space for open dialogue about concerns.",
"Just got my COVID vaccine and feeling great! So grateful for the science and healthcare workers that made this possible.",
"Seeing the numbers decline in COVID cases as vaccine distribution increases is such a relief. #VaccinesWork",
"I was hesitant at first, but after doing my research, I'm fully on board with the COVID vaccines. Protecting my community feels good!",
"Required vaccines for school: Parents and guardians of children with school exclusion letters now have an..."
])
labels = np.array([1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
vectorized_tweet = vectorizer.transform(tweets)

# print(clf.coef_)
pred = clf.predict(vectorized_tweet)
print(pred.tolist())