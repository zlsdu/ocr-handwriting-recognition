# coding=utf-8
__author__ = 'lenovo'
import os
from io import BytesIO
from PIL import Image
import requests as req
import xlrd

ma={'：':':','？':'?','、':',','‘':'\'','’':'\'','，':',','。':'.',"！":'!','—':'-','”':'\"'}
specialPunctuationSet=set([',','.','?',':','\'','\"','：','？','、','‘','，','。','！','”','—'])
def rectifyLine(line):
    result=""
    i=0
    while i<len(line):
        if line[i] in ma:
            result+=ma[line[i]]
        elif line[i]!=' ' or i==len(line)-1 or line[i+1] not in specialPunctuationSet:
            result+=line[i]
        i+=1
    return result

nCorrect=0
for p in range(3,4):
    workbook = xlrd.open_workbook('1101_score_1_%d_result.xlsx'%(p+1))
    booksheet = workbook.sheet_by_index(0)
    for i in range(300):
        text=booksheet.cell_value(i+1,6).replace("\u2028","\n").strip("\n")
        url=booksheet.cell_value(i+1,p>0 and 10 or 11)
        if len(url)<3:
            continue
        response = req.get(url)
        goldLines=text.split("\n")
        j=0
        while j<len(goldLines):
            if len(goldLines[j])==0:
                del goldLines[j]
            else:
                j+=1
        for j in range(len(goldLines)):
            goldLines[j]=rectifyLine(goldLines[j].strip())
        text="\n".join(goldLines)
        im = Image.open(BytesIO(response.content))
        im.save("test_img%s%d_%s.png"%(os.sep,p+1,booksheet.cell_value(i+1,1)))
        fout=open("test_txt%s%d_%s.txt"%(os.sep,p+1,booksheet.cell_value(i+1,1)),"w")
        fout.write(text)
        fout.close()
        print("output %d images and texts"%(i+1))