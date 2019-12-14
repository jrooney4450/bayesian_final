

def main():
    word = "Hello, friend"
    try:
        print(word.split(",")[2])
    except:
        print("oops")
if __name__ == "__main__":
    main()