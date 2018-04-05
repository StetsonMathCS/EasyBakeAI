from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import os

app = ClarifaiApp(api_key=os.environ["CLARIFAI_KEY"])

# get the general model
#model = app.models.get("aaa03c23b3724a16a56b629203edc62c") # general
#model = app.models.get("bd367be194cf45149e75f01d59f77ba7") # food
model = app.models.get("c386b7a870114f4a87477c0824499348") # wedding


# predict with the model
image = ClImage(file_obj=open('../../images/josh-pie.jpg', 'rb'))
result = model.predict([image])
print(result)

for concept in result['outputs'][0]['data']['concepts']:
    print("%16s %.4f" % (concept['name'], concept['value']))
