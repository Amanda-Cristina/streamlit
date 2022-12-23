import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from gui import main as gui
from main import main as segmentation_app


def dpills_name3():
    x = """
    <style>

    @import url('https://fonts.googleapis.com/css?family=Roboto:700');

    #container body {
    margin:0px;
    font-family:'Roboto';
    text-align:center;
    }

    #container {
    color:#999;
    text-transform: uppercase;
    font-size:72px;
    font-weight:bold;
    padding-top:200px;  
    position:fixed;
    width:100%;
    bottom:45%;
    display:block;
    }

    #flip {
    height:90px;
    overflow:hidden;
    }

    #flip > div > div {
    color:#fff;
    padding:4px 12px;
    height:80px;
    margin-bottom:45px;
    display:inline-block;
    }

    #flip div:first-child {
    animation: show 5s linear infinite;
    }

    #flip div div {
    background:#42c58a;
    }
    #flip div:first-child div {
    background:#4ec7f3;
    }
    #flip div:last-child div {
    background:#DC143C;
    }

    @keyframes show {
    0% {margin-top:-300px;}
    5% {margin-top:-240px;}
    33% {margin-top:-240px;}
    38% {margin-top:-120px;}
    66% {margin-top:-120px;}
    71% {margin-top:0px;}
    99.99% {margin-top:0px;}
    100% {margin-top:-270px;}
    }

    #container p {
    position:fixed;
    width:100%;
    bottom:30px;
    font-size:12px;
    color:#999;
    margin-top:200px;
    }
    </style>
    <div id=container>
    DP6 
    <div id=flip>
        <div><div>Makes It Possible</div></div>
        <div><div>Your Data Science</div></div>
        <div><div>You Imagine</div></div>
    </div>
    DPills
    </div>
    """
    return x

im = Image.open("imgs/logo_dp6_2.png")
st.set_page_config(
        page_title="Dpills",
        page_icon=im,
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items=None
    )

st.write("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Ubuntu&display=swap');
    html, body, [class*="css"]  {
    font-family: 'Ubuntu';
    }
    </style>
    """, unsafe_allow_html=True)

banner = st.container()
paginas = "<Escolha uma pill>"

with st.sidebar:
    st.write(
        """
        <p style='text-align:center' width='20px' ><img src='https://media-exp1.licdn.com/dms/image/C560BAQFmYhiD6h_39Q/company-logo_200_200/0/1636750457565?e=2147483647&v=beta&t=W9gmVdeNkA-jwA8HLoWHrKqk5uce9Z_BHn7FJwYqfDQ' alt='Logo'></p>
        """, unsafe_allow_html=True
    )
    
    st.write(f"""<p>Logged as </p>""", unsafe_allow_html=True)
    
    paginas = st.selectbox("Qual pill você deseja?", options=("Escolha uma pill", "rfv", "classificacao", "forecasting", "segmentation"))
if paginas == "segmentation":
    gui(exc_function=segmentation_app, config_path="configs.yaml")

if paginas == "Escolha uma pill":
    banner.write(dpills_name3(), unsafe_allow_html=True)
