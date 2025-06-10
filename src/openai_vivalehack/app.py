from davia import Davia

app = Davia()


@app.get("/")
def read_root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    app.run(browser=False)
