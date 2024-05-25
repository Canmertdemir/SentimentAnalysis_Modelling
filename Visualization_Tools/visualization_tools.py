import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud


def word_frequencies_visualization(dataframe):

    frequencies = dataframe['Review'].apply(lambda x: pd.value_counts(x.split(' '))).sum(axis=0).reset_index()
    frequencies.columns = ['Words', 'Frequencies']
    frequencies = frequencies.sort_values('Frequencies', ascending=False)


    frequencies = frequencies[frequencies['Frequencies'] > 500]

    if not frequencies.empty:
        frequencies.plot.bar(x='Words', y='Frequencies')
        plt.savefig('frequencies_word.png')
        plt.show()
    else:
        print("No words with frequency greater than 500.")

def create_world_cloud(dataframe):
    pure_text = ' '.join(word for word in dataframe['Review'].values)
    world_cloud = WordCloud().generate(pure_text)
    plt.imshow(world_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    world_cloud.to_file("wordcloud.png")

