from typing import Union
from fastapi import FastAPI, File, UploadFile
from scripts import helpers, segmentation
import time

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f'{process_time:0.4f} sec')
    return response

@app.get("/")
def read_root():
    return {"response": "Server is running"}


@app.post("/recognize/{tag_width}")
async def recognize(tag_width : float ,uploaded_file: UploadFile = File(...)):
    
    # generate a temporal filename
    filename = f"{helpers.generate_random_file_name()}@{uploaded_file.filename}"
    file_location = f"data/{filename}"
    
    # save file temporally
    helpers.saveUploadfile(file_location, uploaded_file)

    # run model on saved image
    dbh = segmentation.getTreeDBH(file_location, tag_width)

    helpers.removefile(file_location)
    # upload file to s3 and delete from local drive in the background
    #background_task.add_task(uploadfileToS3, file_location, filename, text)

    return {
            "dbh": dbh
        }