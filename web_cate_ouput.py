import pymysql
import time
import json
import datetime
import requests
import csv
import re
import pandas as pd
import numpy as np


#DART01 server

def getCategory(uri):
    """ return: empty str '' => failed to get category
        PS: "Unclassified" is valid (returned by FortiGuard)
    """
    API = 'http://www.fortiguard.com/webfilter?q='
    cat = ''
    try:
        # import requests
        r = requests.get(API + uri, timeout=60)
        # print r.content[:200]
        # print len(r.content)
        html = r.content

        # import re
        result = re.search(b'Category:(.*)" />', html)  # previous version
        if result is None:  # 2nd chance
            result = re.search(b':(.*)</h4>', html)  # since 2015-08-31

        cat = result.group(1).strip()

    except:  # more exceptions & errors -- ref: http://docs.python-requests.org/en/latest/user/quickstart/
        #print "getCategory error: " + uri
        import traceback
        traceback.print_exc()
        # f = open('e:\\tmp\\error.html', 'w+')
        # f.write(uri + ":\n" + html)
        # f.close()


    ## based on urllib2:
    ## Q: why not urllib2? (http://stackoverflow.com/questions/2018026/should-i-use-urllib-or-urllib2-or-requests)
    #     try:
    #         # import urllib2
    #         response = urllib2.urlopen(API+uri)
    #         html = response.read()  # print html
    #         # print html[:200]
    #         print len(html)
    #
    #         # import re
    #         result = re.search('Category:(.*)</h3> <a', html)
    #         cat = result.group(1).strip()
    #     except AttributeError:
    #         print "Invalid Response: " + uri
    #         print html[:1000]
    # #            f = open('e:\\tmp\\error.html', 'w+')
    # #            f.write(uri+":\n" + html)
    # #            f.close()
    #     except httplib.BadStatusLine:
    #         print "BadStatusLine(httplib): " + uri
    #     except: # e.g. URLError
    #         print "more getCategory error: " + uri
    #         import traceback
    #         traceback.print_exc()

    # if cat is None or cat == 'Unclassified':
    if cat is None:
        cat = ''

    # cat = 'TEST'

    #print "getCategory: " + uri + ":" + cat
    if len(cat) > 50:
        # raise ValueError
        print("getCategory warning: len(cat) > 50 -- " + cat)
        # cat = ''
    return cat

def load_csv():
    data = pd.read_csv('web_cate_DB.csv')
    __cate__ = {}
    for i in range(0, len(data), 1):
        __cate__[data['url'][i]] = data['cate'][i]
    return __cate__

def main():
    #variable
    __cate__ = {}

    print("csv loading...")
    __cate__ = load_csv()
    print("loading complete")
    for cate in __cate__:
        if isinstance(__cate__[cate], basestring):
            pass

        else:
            print("fetching cate")
            __cate__[cate] = getCategory(cate)

    with open('web_cate_DB_v2.csv','w') as f:
        wr = csv.writer(f)
        tmp_title = ["url", "cate"]
        wr.writerow(tmp_title)
        for url in __cate__:
            values = []
            values.append(url)
            values.append(__cate__[url])
            wr.writerow(values)




if __name__ == '__main__':
    main()
