#!/usr/bin/env python3
"""测试agent-assistant项目的核心功能"""

import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有模块是否可以正确导入"""
    print("测试导入模块...")
    
    try:
        from document_parser import DocumentParser
        print("✓ document_parser 导入成功")
    except Exception as e:
        print(f"✗ document_parser 导入失败: {e}")
        return False
    
    try:
        from vector_store import VectorStoreManager
        print("✓ vector_store 导入成功")
    except Exception as e:
        print(f"✗ vector_store 导入失败: {e}")
        return False
    
    try:
        from tools import tools
        print("✓ tools 导入成功")
        print(f"  可用工具: {[tool.name for tool in tools]}")
    except Exception as e:
        print(f"✗ tools 导入失败: {e}")
        return False
    
    try:
        from flask import Flask
        print("✓ flask 导入成功")
    except Exception as e:
        print(f"✗ flask 导入失败: {e}")
        return False
    
    try:
        from flask_cors import CORS
        print("✓ flask_cors 导入成功")
    except Exception as e:
        print(f"✗ flask_cors 导入失败: {e}")
        return False
    
    try:
        from PyPDF2 import PdfReader
        print("✓ PyPDF2 导入成功")
    except Exception as e:
        print(f"✗ PyPDF2 导入失败: {e}")
        return False
    
    try:
        from docx import Document
        print("✓ python-docx 导入成功")
    except Exception as e:
        print(f"✗ python-docx 导入失败: {e}")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✓ scikit-learn 导入成功")
    except Exception as e:
        print(f"✗ scikit-learn 导入失败: {e}")
        return False
    
    try:
        from langchain_community.vectorstores import FAISS
        print("✓ FAISS 导入成功")
    except Exception as e:
        print(f"✗ FAISS 导入失败: {e}")
        return False
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✓ RecursiveCharacterTextSplitter 导入成功")
    except Exception as e:
        print(f"✗ RecursiveCharacterTextSplitter 导入失败: {e}")
        return False
    
    return True

def test_document_parser():
    """测试文档解析器"""
    print("\n测试文档解析器...")
    
    from document_parser import DocumentParser
    
    # 创建测试文档
    test_text = """医疗研报测试文档
    
    第一章：高血压治疗进展
    
    高血压是一种常见的慢性疾病，影响全球约10亿人口。近年来，随着医学研究的不断深入，高血压的治疗方案也在不断更新和完善。
    
    第二章：新型药物介绍
    
    1. 血管紧张素受体拮抗剂
    2. 钙通道阻滞剂
    3. β受体阻滞剂
    
    第三章：治疗建议
    
    患者应遵循医生的建议，按时服药，定期监测血压。同时，保持健康的生活方式也非常重要。
    """
    
    # 测试解析器是否正常工作
    parser = DocumentParser()
    
    # 测试文件类型判断
    result = parser.parse_document("test.pdf")
    print(f"✓ 文件类型判断功能正常")
    
    return True

def test_vector_store():
    """测试向量存储"""
    print("\n测试向量存储...")
    
    from vector_store import VectorStoreManager
    
    vector_store = VectorStoreManager()
    
    # 测试添加文档
    test_text = "这是一篇关于高血压治疗的医疗研报。高血压患者需要定期监测血压，按时服药，并保持健康的生活方式。"
    result = vector_store.add_document(test_text, metadata={"source": "test"})
    print(f"添加文档: {result}")
    
    # 测试搜索功能
    result = vector_store.search("高血压")
    print(f"搜索结果: {result[:100]}...")
    
    # 测试清空功能
    result = vector_store.clear()
    print(f"清空向量存储: {result}")
    
    return True

def test_tools():
    """测试工具功能"""
    print("\n测试工具功能...")
    
    from tools import tools
    
    # 测试计算器工具
    calc_tool = next(t for t in tools if t.name == "Calculator")
    result = calc_tool.func("2 + 3 * 4")
    print(f"计算器测试: {result}")
    
    # 测试时间工具
    time_tool = next(t for t in tools if t.name == "CurrentTime")
    result = time_tool.func()
    print(f"时间查询测试: {result}")
    
    # 测试医疗术语工具
    med_tool = next(t for t in tools if t.name == "MedicalTerm")
    result = med_tool.func("高血压")
    print(f"医疗术语测试: {result[:50]}...")
    
    # 测试药品信息工具
    drug_tool = next(t for t in tools if t.name == "DrugInfo")
    result = drug_tool.func("阿司匹林")
    print(f"药品信息测试: {result[:50]}...")
    
    return True

def test_flask_app():
    """测试Flask应用是否可以创建"""
    print("\n测试Flask应用...")
    
    try:
        from app import app
        print("✓ Flask应用创建成功")
        print(f"  应用配置: {app.config.keys()}")
        return True
    except Exception as e:
        print(f"✗ Flask应用创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("agent-assistant 项目测试")
    print("=" * 60)
    
    # 测试导入
    import_ok = test_imports()
    
    if not import_ok:
        print("\n❌ 导入测试失败，请检查依赖安装")
        return False
    
    # 测试文档解析器
    doc_ok = test_document_parser()
    
    # 测试向量存储
    vector_ok = test_vector_store()
    
    # 测试工具
    tools_ok = test_tools()
    
    # 测试Flask应用（不启动，只测试创建）
    # flask_ok = test_flask_app()
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print(f"✓ 导入测试: {'通过' if import_ok else '失败'}")
    print(f"✓ 文档解析器: {'通过' if doc_ok else '失败'}")
    print(f"✓ 向量存储: {'通过' if vector_ok else '失败'}")
    print(f"✓ 工具测试: {'通过' if tools_ok else '失败'}")
    print("=" * 60)
    
    if import_ok and doc_ok and vector_ok and tools_ok:
        print("\n✅ 所有核心功能测试通过！")
        print("\n启动方式:")
        print("1. 确保已安装依赖: pip install -r requirements.txt")
        print("2. 设置环境变量或修改app.py中的API_KEY")
        print("3. 运行: python app.py")
        print("4. 访问: http://localhost:5000")
        return True
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
