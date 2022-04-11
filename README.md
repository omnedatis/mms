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
  
當系統初始啟動時, 會偵測資料夾下是否存在 `_local` 資料夾, 若不存在則會啟動初始化  
初始化時, 會執行復原的動作, 這是假設系統經過意外崩潰重新建立後的狀況  
執行復原時, 若資料庫中存在現象資料, 但不存在市場資料時, 這時候會對系統中的模型進行 **偽訓練**  
    在**偽訓練**的情況下, 並不會真正於 python server 中建立模型, 但仍會在資料庫中標記建立完成  
訓練模型時所使用的市場, 若是存在一筆以上的市場時, 採用**烙印當前全市場**進行訓練, 並**永久保存**  
若不存在任何市場, 則採用 **偽訓練** 的方式, 舉例來說:  
當初始化資料庫時, 資料表中沒有任何資料, 接著將部分現象與模型資訊導入後觸發系統初始化  
此時由於系統中不存在任何市場資料, 因此會使用**偽訓練**建立模型  
這時, 若有人於資料庫中新增**一個市場**, 並觸發批次, 這時候系統就會使用**這一個市場**進行所有模型的建立, 並於未來每日都使用該模型進行預測  
因此即使未來增加了 1000 個市場, 這些初始化時建立的模型也只會用當初建立時所使用的**一個市場**進行所有市場的預測  

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

