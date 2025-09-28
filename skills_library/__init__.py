"""
Skills Library for Robotic Manipulation

A modular library of reusable manipulation skills including:
- Push: Push objects along a surface
- Slide: Slide objects to target positions
- Stack: Stack objects on top of each other
- Pick & Place: Basic grasping and placement
"""

from .base_skill import BaseSkill, SkillResult, SkillStatus
from .skill_registry import SkillRegistry, skill_registry
from .push_skill import PushSkill
from .slide_skill import SlideSkill
from .stack_skill import StackSkill
from .pick_place_skill import PickPlaceSkill

__all__ = [
    'BaseSkill',
    'SkillResult', 
    'SkillStatus',
    'SkillRegistry',
    'skill_registry',
    'PushSkill',
    'SlideSkill',
    'StackSkill',
    'PickPlaceSkill'
]

# Auto-register default skills
def register_default_skills():
    """Register all default skills in the library"""
    skill_registry.register('push', PushSkill)
    skill_registry.register('slide', SlideSkill)
    skill_registry.register('stack', StackSkill)
    skill_registry.register('pick_place', PickPlaceSkill)

# Register on import
register_default_skills()