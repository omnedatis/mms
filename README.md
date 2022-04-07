# mimosa-ai

## Quickstart
若要執行 Server , 可以下列指令執行
```
cd workspace_path
conda activate py_cal
python server.py [-bl] [-ml] [-md dev]
```
bl: 啟動但不於啟動當下執行一次批次  
ml: 不啟動 Server
md: 執行環境參數 預設為 dev

## Overview
這個專案主要提供 Mimosa 的主網站後台運算服務, 其中包含
 - Python 每日計算每個現象對於每個市場的發生與否
 - Python 每日計算每個現象每個市場的未來各天期報酬的各個統計資訊: 包含 總現象發生次數, 未發生次數, 現象發生後上漲次數, 現象發生後持平次數, 現象發生後下跌次數
 - Python 啟動後的前次執行狀態還原機制, 當 Python Server 意外中斷並重啟後需還原前次最後執行狀態, 並將其重啟並完成
 - Python Server 的各項功能, 包含: 新增觀點, 刪除觀點

## Installation
安裝的方法需搭配 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 進行安裝  
Mimoconda 安裝完成後, 使用以下指令建立 Pyhton 環境  
```
cd workspace_path
conda env create --file requirements.yml
```
預設建立的環境名稱為`py_cal`, 若有需要則可以更改  

## Copyright and Licenses
Code and documentation copyright 2022 Softbi, Inc.

