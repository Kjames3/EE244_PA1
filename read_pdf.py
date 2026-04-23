import pypdf
import sys

def read_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(text)

if __name__ == "__main__":
    read_pdf(sys.argv[1])
