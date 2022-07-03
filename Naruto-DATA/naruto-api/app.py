from fastapi import FastAPI,Path
from pydantic import BaseModel
import csv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)  


@app.get("/{episode}")
def getNaruto(episode:str):
    narutodict = {}
    with open("naruto.csv",'r') as file:
        reader = csv.reader(file)
        for column in reader:
            if column[1] == episode:
                title = column[2]
                type = column[3]
                year = column[4]
                rate = column[5]
                votes = column[6]
                saga = column[7]
                airdate = column[8]
                narutodict = {"Title":title,"Type":type,"Year_Launched":year,"Rate":rate,"Votes":votes,"Sage":saga,"AirDate":airdate}
    return narutodict