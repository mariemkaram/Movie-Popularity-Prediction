import pandas as pd
import fasttext
lang_model=fasttext.load_model('C:/Users/USER/anaconda3/envs\ml/Lib/site-packages/fasttext/lid.176.bin')
from googletrans import Translator

translator=Translator()
mydata = pd.read_csv('E:\pattern\movies-regression\movies-regression-dataset.csv')
for i in range(len(mydata.tagline)):
    if isinstance(mydata.tagline[i], str) :
        #translate_text=translator.translate(mydata.tagline[i],dest='en')
        lang=lang_model.predict(mydata.tagline[i],k=1)[0][0][9:11]
        if lang !='en':
            translated=translator.translate(mydata.tagline[i],dest='en')
            mydata.tagline[i]=translated.text
            print(i)
            print(translated.text)
print(mydata.tagline)