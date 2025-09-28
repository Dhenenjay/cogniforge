
class CogniforgePhysics:
    """
    Physics wrapper for Cogniforge project that abstracts the physics engine.
    Can work with PyBullet (3D) or PyMunk (2D) depending on availability.
    """
    
    def __init__(self, use_3d=True):
        self.use_3d = use_3d
        self.engine = None
        self.space = None
        
        if use_3d:
            try:
                import pybullet as p
                self.engine = 'pybullet'
                self.p = p
                print("Using PyBullet for 3D physics")
            except ImportError:
                print("PyBullet not available, falling back to 2D physics")
                self.use_3d = False
        
        if not self.use_3d:
            try:
                import pymunk
                self.engine = 'pymunk'
                self.pymunk = pymunk
                self.space = pymunk.Space()
                self.space.gravity = (0, -981)
                print("Using PyMunk for 2D physics")
            except ImportError:
                raise ImportError("No physics engine available!")
    
    def create_ground(self):
        """Create a ground plane/line."""
        if self.engine == 'pybullet':
            plane_id = self.p.createCollisionShape(self.p.GEOM_PLANE)
            return self.p.createMultiBody(0, plane_id)
        elif self.engine == 'pymunk':
            ground = self.pymunk.Segment(self.space.static_body, (0, 0), (1000, 0), 5)
            ground.friction = 1.0
            self.space.add(ground)
            return ground
    
    def create_box(self, position, size=1.0, mass=1.0):
        """Create a box object."""
        if self.engine == 'pybullet':
            half_size = size / 2.0
            collision = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[half_size]*3)
            visual = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=[half_size]*3)
            return self.p.createMultiBody(baseMass=mass,
                                        baseCollisionShapeIndex=collision,
                                        baseVisualShapeIndex=visual,
                                        basePosition=position)
        elif self.engine == 'pymunk':
            moment = self.pymunk.moment_for_box(mass, (size*100, size*100))
            body = self.pymunk.Body(mass, moment)
            body.position = position[0]*100, position[1]*100  # Convert to pymunk scale
            shape = self.pymunk.Poly.create_box(body, (size*100, size*100))
            self.space.add(body, shape)
            return body
    
    def step_simulation(self, timestep=1/60.0):
        """Step the physics simulation."""
        if self.engine == 'pybullet':
            self.p.stepSimulation()
        elif self.engine == 'pymunk':
            self.space.step(timestep)
    
    def get_position(self, obj):
        """Get object position."""
        if self.engine == 'pybullet':
            pos, _ = self.p.getBasePositionAndOrientation(obj)
            return pos
        elif self.engine == 'pymunk':
            return obj.position.x/100, obj.position.y/100  # Convert back from pymunk scale
    