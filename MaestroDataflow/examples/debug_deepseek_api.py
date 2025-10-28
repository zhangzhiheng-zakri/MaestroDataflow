#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek API调试脚本
专门用于分析400 Bad Request错误
"""

import requests
import json


def test_deepseek_api_direct():
    """直接测试DeepSeek API"""
    print("=== 直接测试DeepSeek API ===")
    
    api_key = "sk-e987d89ccdbe46c6948112314096b038"
    api_url = "https://api.deepseek.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 测试不同的请求体格式
    test_cases = [
        {
            "name": "标准格式",
            "data": {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 10,
                "temperature": 0.3
            }
        },
        {
            "name": "简化格式",
            "data": {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }
        },
        {
            "name": "不同模型名",
            "data": {
                "model": "deepseek-coder",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 10
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 测试: {test_case['name']}")
        print(f"📝 请求体: {json.dumps(test_case['data'], indent=2, ensure_ascii=False)}")
        
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=test_case['data'],
                timeout=30
            )
            
            print(f"📡 状态码: {response.status_code}")
            print(f"📡 响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 成功: {result}")
            else:
                print(f"❌ 失败: {response.text}")
                
        except Exception as e:
            print(f"❌ 异常: {e}")


def test_api_info():
    """测试API基本信息"""
    print("\n=== 测试API基本信息 ===")
    
    # 测试根路径
    try:
        response = requests.get("https://api.deepseek.com", timeout=10)
        print(f"🌐 根路径状态码: {response.status_code}")
        print(f"🌐 根路径响应: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ 根路径测试失败: {e}")
    
    # 测试模型列表（如果支持）
    try:
        api_key = "sk-e987d89ccdbe46c6948112314096b038"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        response = requests.get(
            "https://api.deepseek.com/v1/models",
            headers=headers,
            timeout=10
        )
        print(f"📋 模型列表状态码: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"📋 可用模型: {models}")
        else:
            print(f"📋 模型列表响应: {response.text}")
    except Exception as e:
        print(f"❌ 模型列表测试失败: {e}")


def analyze_request_format():
    """分析请求格式问题"""
    print("\n=== 分析请求格式 ===")
    
    # 检查API密钥格式
    api_key = "sk-e987d89ccdbe46c6948112314096b038"
    print(f"🔑 API密钥长度: {len(api_key)}")
    print(f"🔑 API密钥格式: {'✅ 正确' if api_key.startswith('sk-') else '❌ 错误'}")
    
    # 检查URL格式
    api_url = "https://api.deepseek.com/v1/chat/completions"
    print(f"🌐 API URL: {api_url}")
    
    # 检查请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    print(f"📋 请求头: {json.dumps(headers, indent=2, ensure_ascii=False)}")


def main():
    """主函数"""
    print("🔍 DeepSeek API调试开始")
    print("=" * 50)
    
    # 分析请求格式
    analyze_request_format()
    
    # 测试API基本信息
    test_api_info()
    
    # 直接测试API
    test_deepseek_api_direct()
    
    print("\n🔍 调试完成")


if __name__ == "__main__":
    main()