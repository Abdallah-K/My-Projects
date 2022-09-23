from fastapi import FastAPI
from fastapi import UploadFile,File
import tensorflow.keras
import numpy as np 
import cv2
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)  



model = tensorflow.keras.models.load_model('flower.h5')

def load_image_into_numpy_array(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/flowers")
async def predict_image(file : UploadFile = File(...)):
    img = load_image_into_numpy_array(await file.read())
    img = cv2.resize(img,(180,180))

        
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    name = np.argmax(score)
    name = name.tolist()
    if name == 0:
        flower = "Daisy"
        overview = "Daisy is a feminine given name. The flower name comes from the Old English word dægeseage, meaning days eye. The name Daisy is therefore ultimately derived from this source. Daisy is also a nickname for Margaret, used because Marguerite, the French version of that name, is also a French name for the oxeye daisyDaisy was the 169th most popular name for girls born in the United States in 2018. It was in steady use for American girls throughout the 20th century and was ranked among the top 200 names for girls between 1900 and 1940. It declined in popularity between 1960 and 1980, but has been climbing in popularity since the 1980s"
    elif name == 4:
        flower = "Tulip"
        overview = "Tulips (Tulipa) are a genus of spring-blooming perennial herbaceous bulbiferous geophytes (having bulbs as storage organs). The flowers are usually large, showy and brightly colored, generally red, pink, yellow, or white (usually in warm colors). They often have a different colored blotch at the base of the tepals (petals and sepals, collectively), internally. Because of a degree of variability within the populations, and a long history of cultivation, classification has been complex and controversial. The tulip is a member of the lily family, Liliaceae, along with 14 other genera, where it is most closely related to Amana, Erythronium and Gagea in the tribe Lilieae. There are about 75 species, and these are divided among four subgenera. The name tulip is thought to be derived from a Persian word for turban, which it may have been thought to resemble by those who discovered it."
    elif name == 3:
        flower = "Sunflower"
        overview = " is a genus comprising about 70 species of annual and perennial flowering plants in the daisy family Asteraceae commonly known as sunflowers.[4][5] Except for three South American species, the species of Helianthus are native to North America and Central America. The most popular species of sunflower is the common sunflower (Helianthus annuus), whose round flower heads in combination with the ligules look like the Sun.[6] This and other species, notably Jerusalem artichoke (H. tuberosus), are cultivated in temperate regions and some tropical regions as food crops for humans, cattle, and poultry, and as ornamental plants.[7] The species H. annuus typically grows during the summer and into early fall, with the peak growth season being mid-summer."
    elif name == 2:
        flower = "Rose"
        overview = "A rose is a woody perennial flowering plant of the genus Rosa, in the family Rosaceae, or the flower it bears. There are over three hundred species and tens of thousands of cultivars.[citation needed] They form a group of plants that can be erect shrubs, climbing, or trailing, with stems that are often armed with sharp prickles.[citation needed] Their flowers vary in size and shape and are usually large and showy, in colours ranging from white through yellows and reds. Most species are native to Asia, with smaller numbers native to Europe, North America, and northwestern Africa.[citation needed] Species, cultivars and hybrids are all widely grown for their beauty and often are fragrant. Roses have acquired cultural significance in many societies. Rose plants range in size from compact, miniature roses, to climbers that can reach seven meters in height.[citation needed] Different species hybridize easily, and this has been used in the development of the wide range of garden roses."
    elif name == 1:
        flower = "Danedelion"
        overview = "Taraxacum  is a large genus of flowering plants in the family Asteraceae, which consists of species commonly known as dandelions. The scientific and hobby study of the genus is known as taraxacology.[3] The genus is native to Eurasia and North America, but the two most commonplace species worldwide, T. officinale (the common dandelion) and T. erythrospermum (the red-seeded dandelion), were introduced into North America from Europe and now propagate as wildflowers. Both species are edible in their entirety. The common name dandelion is also given to specific members of the genus.Like other members of the family Asteraceae, they have very small flowers collected together into a composite flower head. Each single flower in a head is called a floret. In part due to their abundance, along with being a generalist species, dandelions are one of the most vital early spring nectar sources for a wide host of pollinators."

    return {"Prediction":flower,"Overview":overview}