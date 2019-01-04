# coding=utf-8
__author__ = 'lenovo'
import os
from io import BytesIO
from PIL import Image
import numpy as np
import requests as req
import xlrd
import re

ma={'：':':','？':'?','、':',','‘':'\'','’':'\'','，':',','。':'.',"！":'!','—':'-','”':'\"'}
specialPunctuationSet=set([',','.','?',':','!','\'','\"','：','？','、','‘','，','。','！','”'])
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

def countMargin(v,minSum,direction=True):
    if direction:
        for i in range(len(v)):
            if v[i]>minSum:
                return i
        return len(v)
    for i in range(len(v)-1,-1,-1):
        if v[i]>minSum:
            return len(v)-i-1
    return len(v)

def splitLine(seg,dataSum,h,maxHeight):
    i=0
    while i<len(seg)-1:
        if seg[i+1]-seg[i]<maxHeight:
            i+=1
            continue
        x=countMargin(dataSum[seg[i]:],3,True)
        y=countMargin(dataSum[:seg[i+1]],3,False)
        if seg[i+1]-seg[i]-x-y<maxHeight:
            i+=1
            continue
        idx=dataSum[seg[i]+x+h:seg[i+1]-h-y].argmin()+h
        if 0.33<=idx/(seg[i+1]-seg[i]-x-y)<=0.67:
            seg.insert(i+1,dataSum[seg[i]+x+h:seg[i+1]-y-h].argmin()+seg[i]+x+h)
        else:
            i+=1

def getLine(im,data,name,goldLines,upperbound=8,lowerbound=25,threshold=30,h=40,minHeight=35,maxHeight=120,beginX=20,endX=-20,beginY=140,endY=1100,merged=True):
    dataSum=data[:,beginX:endX].sum(1)
    lastPosition=beginY
    seg=[]
    flag=True
    cnt=0
    for i in range(beginY,endY):
        if dataSum[i]<=lowerbound:
            flag=True
            if dataSum[i]<=upperbound:
                cnt=0
                continue
        if flag:
            cnt+=1
            if cnt>=threshold:
                lineNo=np.argmin(dataSum[lastPosition:i])+lastPosition if threshold<=i-beginY else beginY
                if not merged or len(seg)==0 or lineNo-seg[-1]-countMargin(dataSum[seg[-1]:],5,True)-countMargin(dataSum[:lineNo],5,False)>minHeight:
                    seg.append(lineNo)
                else:
                    avg1=dataSum[max(0,seg[-1]-1):seg[-1]+2]
                    avg1=avg1.sum()/avg1.shape[0]
                    avg2=dataSum[max(0,lineNo-1):lineNo+2]
                    avg2=avg2.sum()/avg2.shape[0]
                    if avg1>avg2:
                        seg[-1]=lineNo
                lastPosition=i
                flag=False
    lineNo=np.argmin(dataSum[lastPosition:]>10)+lastPosition if threshold<i else beginY
    if not merged or len(seg)==0 or lineNo-seg[-1]-countMargin(dataSum[seg[-1]:],10,True)-countMargin(dataSum[:lineNo],10,False)>minHeight:
        seg.append(lineNo)
    else:
        avg1=dataSum[max(0,seg[-1]-1):seg[-1]+2]
        avg1=avg1.sum()/avg1.shape[0]
        avg2=dataSum[max(0,lineNo-1):lineNo+2]
        avg2=avg2.sum()/avg2.shape[0]
        if avg1>avg2:
            seg[-1]=lineNo
    splitLine(seg,dataSum,h,maxHeight)
    if len(goldLines)==len(seg)-1:
        for i in range(0,len(seg)-1):
            im.crop((0,seg[i]+countMargin(dataSum[seg[i]:],0),im.size[0],seg[i+1]-countMargin(dataSum[:seg[i+1]],0,False))).save("train_img%s%s_%d.png"%(os.sep,name,i+1))
            fout=open("train_txt%s%s_%d.txt"%(os.sep,name,i+1),"w")
            fout.write(goldLines[i].strip())
            fout.close()
    return len(seg)-1



nCorrect=0
beginYPosition=[140,140,140,125]
for p in range(0,3):
    workbook = xlrd.open_workbook('1101_score_1_%d_result.xlsx'%(p+1))
    booksheet = workbook.sheet_by_index(0)
    for i in range(300):
        text=booksheet.cell_value(i+1,6).strip("\n")
        url=booksheet.cell_value(i+1,p>0 and p<4 and 10 or 11)
        if len(url)<3:
            continue
        response = req.get(url)
        im = Image.open(BytesIO(response.content))
        width,height = im.size
        im = im.convert("L")
        data = im.getdata()
        data = 1-np.asarray(data,dtype='float')/255.0
        data = data.reshape(height,width)
        goldLines=re.split("\u2028|\n",text)
        j=0
        while j<len(goldLines):
            goldLines[j]=rectifyLine(goldLines[j])
            if len(goldLines[j])==0:
                del goldLines[j]
            else:
                goldLines[j]=rectifyLine(goldLines[j])
                j+=1
        nSegLines=getLine(im,data,"%d_%s"%(p+1,booksheet.cell_value(i+1,1)),goldLines,beginY=beginYPosition[p])
        nCorrect+=len(goldLines)==nSegLines and 1 or 0
        print(i+1,nCorrect,len(goldLines),nSegLines)