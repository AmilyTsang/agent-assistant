#!/usr/bin/env python3
"""简单测试脚本 - 验证核心功能"""

import sys

def test_basic():
    """测试基础功能"""
    print("测试基础功能...")
    
    # 测试基本Python功能
    try:
        import os
        print("✓ os 模块可用")
    except:
        print("✗ os 模块不可用")
        return False
    
    try:
        import json
        print("✓ json 模块可用")
    except:
        print("✗ json 模块不可用")
        return False
    
    try:
        import numpy as np
        print("✓ numpy 模块可用")
    except Exception as e:
        print(f"✗ numpy 模块不可用: {e}")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✓ sklearn 模块可用")
    except Exception as e:
        print(f"✗ sklearn 模块不可用: {e}")
        return False
    
    return True

def test_project_modules():
    """测试项目模块"""
    print("\n测试项目模块...")
    
    try:
        from document_parser import DocumentParser
        print("✓ document_parser 模块可用")
    except Exception as e:
        print(f"✗ document_parser 模块不可用: {e}")
        return False
    
    try:
        from vector_store import VectorStoreManager, SimpleEmbeddings
        print("✓ vector_store 模块可用")
    except Exception as e:
        print(f"✗ vector_store 模块不可用: {e}")
        return False
    
    try:
        from tools import tools
        print("✓ tools 模块可用")
    except Exception as e:
        print(f"✗ tools 模块不可用: {e}")
        return False
    
    return True

def test_vector_store_functionality():
    """测试向量存储功能"""
    print("\n测试向量存储功能...")
    
    from vector_store import VectorStoreManager
    
    try:
        vs = VectorStoreManager()
        print("✓ 创建向量存储管理器成功")
    except Exception as e:
        print(f"✗ 创建向量存储管理器失败: {e}")
        return False
    
    try:
        result = vs.add_document("测试文档内容")
        print(f"✓ 添加文档成功: {result}")
    except Exception as e:
        print(f"✗ 添加文档失败: {e}")
        return False
    
    try:
        result = vs.search("测试")
        print(f"✓ 搜索功能正常: {result[:50]}...")
    except Exception as e:
        print(f"✗ 搜索功能失败: {e}")
        return False
    
    try:
        result = vs.clear()
        print(f"✓ 清空功能正常: {result}")
    except Exception as e:
        print(f"✗ 清空功能失败: {e}")
        return False
    
    return True

def test_tools_functionality():
    """测试工具功能"""
    print("\n测试工具功能...")
    
    from tools import tools
    
    try:
        # 测试计算器
        calc = next(t for t in tools if t.name == "Calculator")
        result = calc.func("10 + 20")
        print(f"✓ 计算器工具正常: {result}")
    except Exception as e:
        print(f"✗ 计算器工具失败: {e}")
        return False
    
    try:
        # 测试时间查询
        time_tool = next(t for t in tools if t.name == "CurrentTime")
        result = time_tool.func()
        print(f"✓ 时间工具正常: {result[:20]}...")
    except Exception as e:
        print(f"✗ 时间工具失败: {e}")
        return False
    
    try:
        # 测试医疗术语
        med_tool = next(t for t in tools if t.name == "MedicalTerm")
        result = med_tool.func("高血压")
        print(f"✓ 医疗术语工具正常")
    except Exception as e:
        print(f"✗ 医疗术语工具失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("agent-assistant 简单测试")
    print("=" * 50)
    
    # 测试基础功能
    basic_ok = test_basic()
    
    if not basic_ok:
        print("\n❌ 基础功能测试失败")
        return False
    
    # 测试项目模块
    modules_ok = test_project_modules()
    
    if not modules_ok:
        print("\n❌ 项目模块测试失败")
        return False
    
    # 测试向量存储
    vs_ok = test_vector_store_functionality()
    
    # 测试工具
    tools_ok = test_tools_functionality()
    
    print("\n" + "=" * 50)
    print("测试结果:")
    print(f"✓ 基础功能: {'通过' if basic_ok else '失败'}")
    print(f"✓ 项目模块: {'通过' if modules_ok else '失败'}")
    print(f"✓ 向量存储: {'通过' if vs_ok else '失败'}")
    print(f"✓ 工具功能: {'通过' if tools_ok else '失败'}")
    print("=" * 50)
    
    if basic_ok and modules_ok and vs_ok and tools_ok:
        print("\n✅ 所有测试通过！")
        print("\n项目结构正常，可以运行。")
        print("请确保已正确设置API Key后启动。")
        return True
    else:
        print("\n❌ 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
