"""
Moodle Content Extractor
Extracts course content from Moodle via Web Services API

Enterprise-grade implementation with:
- Retry logic and error handling
- Progress tracking
- Multiple content type support
- Rate limiting protection
"""

import os
import requests
import logging
from typing import List, Dict, Optional
from urllib.parse import urljoin
import time
import re

logger = logging.getLogger(__name__)

class MoodleExtractor:
    """
    Extract course content from Moodle using Web Services API
    """
    
    def __init__(self):
        self.base_url = os.getenv("MOODLE_URL", "").rstrip("/")
        self.token = os.getenv("MOODLE_TOKEN", "")
        self.ws_endpoint = f"{self.base_url}/webservice/rest/server.php"
        
        # Extraction settings
        self.extract_pages = os.getenv("MOODLE_EXTRACT_PAGES", "true").lower() == "true"
        self.extract_files = os.getenv("MOODLE_EXTRACT_FILES", "true").lower() == "true"
        self.extract_forums = os.getenv("MOODLE_EXTRACT_FORUMS", "false").lower() == "true"
        self.max_file_size_mb = int(os.getenv("MOODLE_MAX_FILE_SIZE_MB", "50"))
        
        # Validate configuration
        if not self.base_url:
            raise ValueError("MOODLE_URL not configured in environment")
        if not self.token:
            raise ValueError("MOODLE_TOKEN not configured in environment")
        
        logger.info(f"[MOODLE] Initialized extractor for: {self.base_url}")
        logger.info(f"[MOODLE] Extract pages: {self.extract_pages}")
        logger.info(f"[MOODLE] Extract files: {self.extract_files}")
        logger.info(f"[MOODLE] Extract forums: {self.extract_forums}")
    
    def _call_api(self, function: str, params: Dict = None, retry: int = 3) -> Dict:
        """
        Call Moodle Web Services API with retry logic
        
        Args:
            function: Moodle web service function name
            params: Additional parameters
            retry: Number of retries on failure
        
        Returns:
            API response as dict
        """
        if params is None:
            params = {}
        
        payload = {
            "wstoken": self.token,
            "wsfunction": function,
            "moodlewsrestformat": "json",
            **params
        }
        
        for attempt in range(retry):
            try:
                logger.debug(f"[MOODLE API] Calling {function} (attempt {attempt + 1}/{retry})")
                
                response = requests.post(
                    self.ws_endpoint,
                    data=payload,
                    timeout=30
                )
                
                response.raise_for_status()
                data = response.json()
                
                # Check for Moodle error response
                if isinstance(data, dict) and "exception" in data:
                    error_msg = data.get("message", "Unknown Moodle error")
                    logger.error(f"[MOODLE API ERROR] {function}: {error_msg}")
                    raise ValueError(f"Moodle API error: {error_msg}")
                
                logger.debug(f"[MOODLE API] ✓ {function} successful")
                return data
                
            except requests.exceptions.Timeout:
                logger.warning(f"[MOODLE API] Timeout on {function} (attempt {attempt + 1})")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise
                
            except requests.exceptions.RequestException as e:
                logger.error(f"[MOODLE API ERROR] {function}: {e}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        raise RuntimeError(f"Failed to call {function} after {retry} attempts")
    
    def get_course_info(self, course_id: int) -> Dict:
        """
        Get basic course information
        
        Args:
            course_id: Moodle course ID
        
        Returns:
            Course information dict
        """
        logger.info(f"[MOODLE] Fetching course info for course_id={course_id}")
        
        courses = self._call_api(
            "core_course_get_courses",
            {"options[ids][0]": course_id}
        )
        
        if not courses or len(courses) == 0:
            raise ValueError(f"Course {course_id} not found")
        
        course = courses[0]
        logger.info(
            f"[MOODLE] ✓ Course found: '{course.get('fullname', 'Unknown')}' "
            f"(ID: {course_id})"
        )
        
        return course
    
    def get_course_contents(self, course_id: int) -> List[Dict]:
        """
        Get all course sections and modules
        
        Args:
            course_id: Moodle course ID
        
        Returns:
            List of course sections with modules
        """
        logger.info(f"[MOODLE] Fetching course contents for course_id={course_id}")
        
        contents = self._call_api(
            "core_course_get_contents",
            {"courseid": course_id}
        )
        
        if not contents:
            logger.warning(f"[MOODLE] No contents found for course {course_id}")
            return []
        
        logger.info(f"[MOODLE] ✓ Found {len(contents)} sections")
        
        # Count total modules
        total_modules = sum(len(section.get("modules", [])) for section in contents)
        logger.info(f"[MOODLE] ✓ Found {total_modules} modules across all sections")
        
        return contents
    
    def extract_course_documents(self, course_id: int) -> List[Dict]:
        """
        Extract all documents from a course for indexing
        
        Args:
            course_id: Moodle course ID
        
        Returns:
            List of documents with structure:
            {
                "type": "page|file|forum|section",
                "content": "Actual text content",
                "metadata": {
                    "course_id": int,
                    "course_name": str,
                    "section_name": str,
                    "module_name": str,
                    ...
                }
            }
        """
        logger.info("\n" + "="*70)
        logger.info(f"[MOODLE EXTRACT] Starting extraction for course_id={course_id}")
        logger.info("="*70)
        
        documents = []
        
        # Get course info
        try:
            course_info = self.get_course_info(course_id)
            course_name = course_info.get("fullname", f"Course {course_id}")
        except Exception as e:
            logger.error(f"[MOODLE EXTRACT ERROR] Failed to get course info: {e}")
            raise
        
        # Get course contents
        try:
            sections = self.get_course_contents(course_id)
        except Exception as e:
            logger.error(f"[MOODLE EXTRACT ERROR] Failed to get course contents: {e}")
            raise
        
        # Extract content from each section
        for section_idx, section in enumerate(sections):
            section_name = section.get("name", f"Section {section_idx + 1}")
            section_summary = section.get("summary", "")
            
            # Add section summary as a document (if not empty)
            if section_summary and len(section_summary.strip()) > 50:
                # Clean HTML tags
                clean_summary = self._clean_html(section_summary)
                
                if clean_summary.strip():
                    documents.append({
                        "type": "section",
                        "content": f"Section: {section_name}\n\n{clean_summary}",
                        "metadata": {
                            "section_id": section.get("id"),
                            "section_name": section_name,
                            "course_id": course_id,
                            "course_name": course_name,
                            "source": f"Section: {section_name}"
                        }
                    })
                    logger.info(f"[EXTRACT] ✓ Section summary: {section_name}")
            
            # Process modules in this section
            modules = section.get("modules", [])
            for module in modules:
                module_type = module.get("modname", "")
                module_name = module.get("name", "Unnamed Module")
                
                # Extract based on module type
                if module_type == "page" and self.extract_pages:
                    doc = self._extract_page_module(module, course_id, course_name, section_name)
                    if doc:
                        documents.append(doc)
                
                elif module_type == "resource" and self.extract_files:
                    doc = self._extract_resource_module(module, course_id, course_name, section_name)
                    if doc:
                        documents.append(doc)
                
                elif module_type == "url":
                    doc = self._extract_url_module(module, course_id, course_name, section_name)
                    if doc:
                        documents.append(doc)
                
                elif module_type == "label":
                    doc = self._extract_label_module(module, course_id, course_name, section_name)
                    if doc:
                        documents.append(doc)
        
        logger.info("\n" + "="*70)
        logger.info(f"[MOODLE EXTRACT] ✓ Extraction complete!")
        logger.info(f"[MOODLE EXTRACT] Total documents: {len(documents)}")
        
        # Log document type breakdown
        type_counts = {}
        for doc in documents:
            doc_type = doc["type"]
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        for doc_type, count in type_counts.items():
            logger.info(f"[MOODLE EXTRACT]   - {doc_type}: {count}")
        
        logger.info("="*70 + "\n")
        
        return documents
    
    def _extract_page_module(self, module: Dict, course_id: int, course_name: str, section_name: str) -> Optional[Dict]:
        """Extract content from a Page module"""
        module_name = module.get("name", "Unnamed Page")
        
        # Page content is in the 'contents' field
        contents = module.get("contents", [])
        
        if not contents:
            logger.debug(f"[EXTRACT] Skipping empty page: {module_name}")
            return None
        
        # Get the main content (usually first item)
        page_content = ""
        for content in contents:
            if content.get("type") == "content":
                page_content = content.get("content", "")
                break
        
        if not page_content:
            return None
        
        # Clean HTML
        clean_content = self._clean_html(page_content)
        
        if len(clean_content.strip()) < 50:
            logger.debug(f"[EXTRACT] Skipping short page: {module_name}")
            return None
        
        logger.info(f"[EXTRACT] ✓ Page: {module_name} ({len(clean_content)} chars)")
        
        return {
            "type": "page",
            "content": f"Page: {module_name}\n\n{clean_content}",
            "metadata": {
                "module_id": module.get("id"),
                "module_name": module_name,
                "section_name": section_name,
                "course_id": course_id,
                "course_name": course_name,
                "source": f"Page: {module_name}"
            }
        }
    
    def _extract_resource_module(self, module: Dict, course_id: int, course_name: str, section_name: str) -> Optional[Dict]:
        """Extract content from a Resource (file) module"""
        module_name = module.get("name", "Unnamed Resource")
        
        # Resource files are in the 'contents' field
        contents = module.get("contents", [])
        
        if not contents:
            logger.debug(f"[EXTRACT] Skipping empty resource: {module_name}")
            return None
        
        # Get file info
        file_info = contents[0]
        file_url = file_info.get("fileurl", "")
        filename = file_info.get("filename", "")
        filesize = file_info.get("filesize", 0)
        
        # Check file size
        filesize_mb = filesize / (1024 * 1024)
        if filesize_mb > self.max_file_size_mb:
            logger.warning(
                f"[EXTRACT] Skipping large file: {filename} "
                f"({filesize_mb:.1f}MB > {self.max_file_size_mb}MB)"
            )
            return None
        
        # For now, we'll add file metadata
        # TODO: Download and extract text from PDFs, docs, etc.
        logger.info(f"[EXTRACT] ✓ Resource: {module_name} (file: {filename})")
        
        return {
            "type": "file",
            "content": f"Resource: {module_name}\nFilename: {filename}\n\nThis is a file resource. File content extraction can be added in future updates.",
            "metadata": {
                "module_id": module.get("id"),
                "module_name": module_name,
                "section_name": section_name,
                "filename": filename,
                "filesize": filesize,
                "file_url": file_url,
                "course_id": course_id,
                "course_name": course_name,
                "source": f"File: {filename}"
            }
        }
    
    def _extract_url_module(self, module: Dict, course_id: int, course_name: str, section_name: str) -> Optional[Dict]:
        """Extract content from a URL module"""
        module_name = module.get("name", "Unnamed URL")
        description = module.get("description", "")
        
        # URLs have external links
        contents = module.get("contents", [])
        external_url = ""
        
        if contents:
            external_url = contents[0].get("fileurl", "")
        
        clean_desc = self._clean_html(description) if description else ""
        
        content = f"Link: {module_name}\n"
        if external_url:
            content += f"URL: {external_url}\n"
        if clean_desc:
            content += f"\n{clean_desc}"
        
        if len(content.strip()) < 30:
            return None
        
        logger.info(f"[EXTRACT] ✓ URL: {module_name}")
        
        return {
            "type": "url",
            "content": content,
            "metadata": {
                "module_id": module.get("id"),
                "module_name": module_name,
                "section_name": section_name,
                "external_url": external_url,
                "course_id": course_id,
                "course_name": course_name,
                "source": f"Link: {module_name}"
            }
        }
    
    def _extract_label_module(self, module: Dict, course_id: int, course_name: str, section_name: str) -> Optional[Dict]:
        """Extract content from a Label module (inline text/html)"""
        module_name = module.get("name", "Label")
        description = module.get("description", "")
        
        if not description:
            return None
        
        clean_content = self._clean_html(description)
        
        if len(clean_content.strip()) < 50:
            return None
        
        logger.info(f"[EXTRACT] ✓ Label in {section_name}")
        
        return {
            "type": "label",
            "content": clean_content,
            "metadata": {
                "module_id": module.get("id"),
                "section_name": section_name,
                "course_id": course_id,
                "course_name": course_name,
                "source": f"Label in {section_name}"
            }
        }
    
    def _clean_html(self, html: str) -> str:
        """
        Clean HTML tags and decode entities
        
        Args:
            html: HTML string to clean
        
        Returns:
            Clean text without HTML tags
        """
        if not html:
            return ""
        
        # Remove script and style tags with their content
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML comments
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        
        # Remove all HTML tags
        html = re.sub(r'<[^>]+>', '', html)
        
        # Decode common HTML entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&amp;', '&')
        html = html.replace('&quot;', '"')
        html = html.replace('&#39;', "'")
        html = html.replace('&apos;', "'")
        html = html.replace('&mdash;', '—')
        html = html.replace('&ndash;', '–')
        html = html.replace('&hellip;', '...')
        html = html.replace('&copy;', '©')
        html = html.replace('&reg;', '®')
        html = html.replace('&trade;', '™')
        
        # Decode numeric HTML entities
        html = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), html)
        html = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), html)
        
        # Clean up whitespace
        html = re.sub(r'\s+', ' ', html)
        html = re.sub(r'\n\s*\n', '\n\n', html)
        html = html.strip()
        
        return html


# ============================================
# GLOBAL INSTANCE
# ============================================

try:
    moodle_extractor = MoodleExtractor()
    logger.info("[MOODLE] Extractor ready")
except Exception as e:
    logger.error(f"[MOODLE ERROR] Failed to initialize extractor: {e}")
    logger.warning("[MOODLE] Course extraction features will be disabled")
    moodle_extractor = None