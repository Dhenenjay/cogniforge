"""
Skill Registry

Central registry for managing and accessing manipulation skills.
"""

from typing import Dict, Type, Optional, List
from .base_skill import BaseSkill
import logging

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Registry for manipulation skills"""
    
    def __init__(self):
        """Initialize skill registry"""
        self._skills: Dict[str, Type[BaseSkill]] = {}
        self._instances: Dict[str, BaseSkill] = {}
        
        # Auto-register default skills so a fresh registry is immediately usable
        try:
            from .push_skill import PushSkill
            from .slide_skill import SlideSkill
            from .stack_skill import StackSkill
            from .pick_place_skill import PickPlaceSkill
            self.register('push', PushSkill)
            self.register('slide', SlideSkill)
            self.register('stack', StackSkill)
            self.register('pick_place', PickPlaceSkill)
        except Exception as e:
            logger.debug(f"Default skill auto-registration skipped: {e}")
        
    def register(self, name: str, skill_class: Type[BaseSkill]):
        """
        Register a skill class
        
        Args:
            name: Skill name
            skill_class: Skill class (must inherit from BaseSkill)
        """
        if not issubclass(skill_class, BaseSkill):
            raise ValueError(f"{skill_class} must inherit from BaseSkill")
            
        self._skills[name] = skill_class
        logger.info(f"Registered skill: {name}")
        
    def unregister(self, name: str):
        """
        Unregister a skill
        
        Args:
            name: Skill name
        """
        if name in self._skills:
            del self._skills[name]
            if name in self._instances:
                del self._instances[name]
            logger.info(f"Unregistered skill: {name}")
            
    def get_skill_class(self, name: str) -> Optional[Type[BaseSkill]]:
        """
        Get skill class by name
        
        Args:
            name: Skill name
            
        Returns:
            Skill class or None
        """
        return self._skills.get(name)
        
    def get_skill(self, name: str):
        """Backwards-compatible accessor used in tests; returns the skill class if registered."""
        return self.get_skill_class(name)
        
    def create_skill(self, name: str, robot_id: int, 
                    scene_objects: Dict[str, int]) -> Optional[BaseSkill]:
        """
        Create skill instance
        
        Args:
            name: Skill name
            robot_id: Robot ID
            scene_objects: Scene objects dictionary
            
        Returns:
            Skill instance or None
        """
        skill_class = self.get_skill_class(name)
        if skill_class:
            instance = skill_class(robot_id, scene_objects)
            self._instances[name] = instance
            return instance
        return None
        
    def get_skill_instance(self, name: str) -> Optional[BaseSkill]:
        """
        Get existing skill instance
        
        Args:
            name: Skill name
            
        Returns:
            Skill instance or None
        """
        return self._instances.get(name)
        
    def list_skills(self) -> List[str]:
        """
        List all registered skill names
        
        Returns:
            List of skill names
        """
        return list(self._skills.keys())
        
    def clear_instances(self):
        """Clear all skill instances"""
        self._instances.clear()
        
    def __repr__(self) -> str:
        return f"SkillRegistry(skills={self.list_skills()})"


# Global registry instance
skill_registry = SkillRegistry()