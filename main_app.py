"""
抽水试验单孔观测数据求参主程序

MIT License

版权所有 (c) 2025 yanggy1010 (yanggy1010@cumt.edu.cn)

特此免费授予任何获得本软件及相关文档文件（以下简称“软件”）副本的人，不受限制地处理本软件的权限，包括
但不限于使用、复制、修改、合并、发布、分发、再许可和/或出售本软件的副本，并允许接收本软件的人这样做，
但须符合以下条件：

上述版权声明和本许可声明应包含在本软件的所有副本或实质性部分中。

本软件按“原样”提供，不提供任何形式的明示或暗示担保，包括但不限于适销性、特定用途适用性和非侵权性的担保。
在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任负责，无论是在合同、侵权或其他行为中产生的，
还是与本软件或本软件的使用或其他交易有关的。
"""


import streamlit as st

# 设置页面配置
st.set_page_config(page_title="抽水试验单孔数据求参")

# 导入外部模块
from home_page import home_page
from hantush_jacob_fit import hantush_jacob_fit
from inflection import inflection
from jacob_line import jacob_line
from jacob_lstsq import jacob_lstsq

# 页面导航
pages = {
    "程序首页": home_page,
    "配线法": hantush_jacob_fit,
    "直线图解法": jacob_line,
    "拐点法": inflection,
    "Jacab公式最小二乘法": jacob_lstsq,
}

# 初始化当前页面状态
if "current_page" not in st.session_state:
    st.session_state.current_page = "程序首页"

# 页面选择器
selected_page = st.sidebar.selectbox("选择求参方法", list(pages.keys()))
if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.rerun()  # 重新运行以加载新页面

# 显示当前页面
pages[st.session_state.current_page]()
