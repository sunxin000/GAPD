"""
Timeout Detection Module
Provides timeout error detection functionality for SPARQL queries
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TimeoutDetector:
    """
    Detects timeout errors in SPARQL query execution results
    """
    
    # Timeout error indicators (case-insensitive)
    TIMEOUT_INDICATORS = (
        'hyt00',  # ODBC timeout error code
        'cl066',  # Virtuoso timeout error code
        'timeout',
    )
    
    @classmethod
    def is_timeout_error(cls, text: str) -> bool:
        """
        Check if text contains timeout error indicators
        
        Args:
            text: Text to check for timeout errors
            
        Returns:
            True if timeout detected, False otherwise
        """
        if not text:
            return False
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in cls.TIMEOUT_INDICATORS)
    
    @classmethod
    def extract_timeout_info(cls, text: str) -> Optional[str]:
        """
        Extract timeout error information from text
        
        Args:
            text: Text containing potential timeout error
            
        Returns:
            Timeout error message if found, None otherwise
        """
        if not cls.is_timeout_error(text):
            return None
        
        # Try to extract the error message
        text_lines = text.split('\n')
        for line in text_lines:
            line_lower = line.lower()
            for indicator in cls.TIMEOUT_INDICATORS:
                if indicator in line_lower:
                    return line.strip()
        
        return "Timeout error detected"


class TimeoutTracker:
    """
    Tracks timeout occurrences during training
    """
    
    def __init__(self):
        self.per_step = {}  # {step: timeout_count}
        self.total = 0
    
    def record_timeout(self, step: int, sample_id: Optional[int] = None):
        """
        Record a timeout occurrence
        
        Args:
            step: Training step where timeout occurred
            sample_id: Optional sample ID for logging
        """
        if step not in self.per_step:
            self.per_step[step] = 0
        self.per_step[step] += 1
        self.total += 1
        
        if sample_id is not None:
            logger.info(f"[TIMEOUT] Sample {sample_id}, Step {step}: "
                       f"Step timeouts={self.per_step[step]}, Total={self.total}")
        else:
            logger.info(f"[TIMEOUT] Step {step}: "
                       f"Step timeouts={self.per_step[step]}, Total={self.total}")
    
    def get_statistics(self) -> dict:
        """Get timeout statistics"""
        return {
            'per_step': dict(self.per_step),
            'total': self.total,
            'steps_with_timeouts': len(self.per_step),
        }
    
    def reset(self):
        """Reset all statistics"""
        self.per_step = {}
        self.total = 0
        logger.info("[TIMEOUT-TRACKER] Statistics reset")
