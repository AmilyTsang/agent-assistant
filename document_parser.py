import os
from PyPDF2 import PdfReader
from docx import Document

class DocumentParser:
    """文档解析器，支持PDF和Word格式"""
    
    @staticmethod
    def parse_pdf(file_path):
        """解析PDF文件"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            return text
        except Exception as e:
            return f"解析PDF文件时出错: {str(e)}"
    
    @staticmethod
    def parse_docx(file_path):
        """解析Word文件"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"解析Word文件时出错: {str(e)}"
    
    @staticmethod
    def parse_document(file_path):
        """根据文件扩展名解析文档"""
        try:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext == '.pdf':
                return DocumentParser.parse_pdf(file_path)
            elif ext == '.docx':
                return DocumentParser.parse_docx(file_path)
            else:
                return f"不支持的文件格式: {ext}"
        except Exception as e:
            return f"解析文档时出错: {str(e)}"
