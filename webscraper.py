import urllib.parse
import requests
from bs4 import BeautifulSoup

results = ""
def getInfoFromWeb(topic):
    # results = ""
    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(topic)}"
    # url2 = 'https://google.com/search?q=' + topic
    # search_result = requests.get(url2)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    info = soup.find(id="mw-content-text")
    # soup2 =BeautifulSoup(search_result.text, 'html.parser')
    # info2 = soup2.find_all('h3')
    # for result in info2:
    #     print(result.text)
    #     results = " | ".join(result.getText())
    # return info, results
    return info.text

if __name__ == "__main__":
    topic = input('Enter topic to search:')
    info = getInfoFromWeb(topic=topic)
    print("Wiki Results = ", info)

