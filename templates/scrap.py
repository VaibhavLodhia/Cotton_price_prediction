
import requests
from bs4 import BeautifulSoup
page = requests.get("https://www.google.dz/search?q=see")
soup = BeautifulSoup(page.content,features="html.parser")
# print(page)
import re
links = soup.findAll("a")
for link in  soup.find_all("a",href=re.compile("(?<=/url\?q=)(htt.*://.*)")):
    print(re.split(":(?=http)",link["href"].replace("/url?q=","")))