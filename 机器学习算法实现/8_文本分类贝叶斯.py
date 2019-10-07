import jieba
import pandas as pd
import numpy

df_news = pd.read_table('val.txt', names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
df_news = df_news.dropna()
# print(df_news.head())
# print(df_news.shape)

content = df_news.content.values.tolist()
# print(content[0])


content_S = []
for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment) > 1 and current_segment != '\r\n':
        content_S.append(current_segment)

# print(content_S[0])

df_content = pd.DataFrame({'content_S': content_S})
# print(df_content.head())

stopwords = pd.read_csv("stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')


def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words

contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)

df_content = pd.DataFrame({'contents_clean': contents_clean})
# print(df_content.head())

df_all_words = pd.DataFrame({'all_words':all_words})

words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({"count":numpy.size})
words_count = words_count.reset_index().sort_values(by=["count"],ascending=False)
print(words_count)
