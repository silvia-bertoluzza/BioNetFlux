
# How `register_implementation` Works in a Factory Pattern (Python)

This explains how your `@classmethod register_implementation` works, how to add a new implementation in `new_implementation.py`, and when the registration should occur.

---

## 1. What `register_implementation` Actually Does

```python
@classmethod
def register_implementation(
    cls,
    problem_type: str,
    implementation_class: Type[StaticCondensationBase]
):
    cls._implementations[problem_type] = implementation_class
```

This method:

1. **Is a class method**
   - `cls` refers to the *factory class itself*, not an instance.

2. **Stores implementations in a shared registry**
   - The factory likely has:
     ```python
     class StaticCondensationFactory:
         _implementations: dict[str, Type[StaticCondensationBase]] = {}
     ```
   - The register method adds a new entry:
     ```python
     {
         "problem_type": ConcreteImplementationClass
     }
     ```

3. **Enables dynamic instantiation later**
   - The factory probably has:
     ```python
     @classmethod
     def create(cls, problem_type: str, *args, **kwargs):
         impl_class = cls._implementations[problem_type]
         return impl_class(*args, **kwargs)
     ```
   - This allows object creation solely based on a string key.

### Conceptually:
> `register_implementation` tells the factory:  
> “If someone asks for problem type `'X'`, instantiate *this* class.”

---

## 2. What to Do When You Add `new_implementation.py`

Example file:

```python
# new_implementation.py

from factory import StaticCondensationFactory
from base import StaticCondensationBase

class MyNewImplementation(StaticCondensationBase):
    ...
```

You must:

### ✅ (A) Define the class  
Already done by creating `MyNewImplementation`.

### ✅ (B) Register the class with the factory

```python
StaticCondensationFactory.register_implementation(
    "my_problem_type",
    MyNewImplementation
)
```

If you *don’t* do this, the factory has no way to discover your new class.

---

## 3. When Should `register_implementation` Be Called?

### ✅ Correct:
> **Exactly once per implementation, at import time.**

- Registration is *configuration*, not usage.
- The mapping remains for the entire Python process.
- You should **NOT** call it every time you create an object.

---

## 4. The Two Proper Registration Patterns

---

### ✅ Option 1 (Best Practice): Self-Registration in `new_implementation.py`

```python
# new_implementation.py

from factory import StaticCondensationFactory
from base import StaticCondensationBase

class MyNewImplementation(StaticCondensationBase):
    ...

# Register on import
StaticCondensationFactory.register_implementation(
    "my_problem_type",
    MyNewImplementation
)
```

Then ensure this file is imported once:

```python
import new_implementation
```

✅ Registration automatically happens when the module is imported.

---

### ✅ Option 2: Centralized Registration File

```python
# registrations.py

from factory import StaticCondensationFactory
from new_implementation import MyNewImplementation

StaticCondensationFactory.register_implementation(
    "my_problem_type",
    MyNewImplementation
)
```

This works but:
- Requires manual updates.
- Is less modular for plugin-style systems.

---

## 5. What You Should **NOT** Do

❌ **Do NOT register inside runtime logic**
```python
obj = Factory.create(...)
Factory.register_implementation(...)  # ❌ Wrong
```

This:
- Wastes time
- Risks overwriting
- Breaks the factory abstraction

---

❌ **Do NOT forget to import the module**
If `new_implementation.py` is never imported, its registration code is never executed:

```python
import new_implementation  # This is required
```

Without this, the factory will raise a `KeyError`.

---

## 6. Typical Correct Workflow

1. Create `new_implementation.py`:
   ```python
   class MyNewImplementation(StaticCondensationBase):
       ...

   StaticCondensationFactory.register_implementation(
       "my_problem_type",
       MyNewImplementation
   )
   ```

2. Import it at program startup:
   ```python
   import new_implementation
   ```

3. Use it anywhere:
   ```python
   obj = StaticCondensationFactory.create("my_problem_type", args...)
   ```

✅ No further registration calls needed.

---

## 7. Mental Model

Think of the factory registry as a **global phonebook**:

| Action | Meaning |
|--------|---------|
| `register_implementation` | Add a phone number |
| `create()` | Look up and call |
| Register once | Correct |
| Create many times | Correct |
| Re-register repeatedly | Wrong |

---

## 8. Summary

✅ Register each implementation **once**  
✅ Do it **at import time**  
✅ Either let the module self-register or use a centralized registration file  
❌ Never register inside normal execution logic  
❌ Never forget to import the module that performs registration  
