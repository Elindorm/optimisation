
### 1. Аналитическое решение задачи оптимизации

В первой части кода производится аналитическое решение функции Розенброка. Сначала вычисляются частные производные функции, а затем решается система уравнений для нахождения стационарных точек:

```python
import sympy as sp

x, y = sp.symbols('x y')
f = 100 * (y - x**2)**2 + (1 - x)**2

df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)

solutions = sp.solve([df_dx, df_dy], (x, y))
```

### 2. Численное решение с использованием метода наискорейшего спуска

Для численного решения используется метод наискорейшего спуска. Функция Розенброка определяется и её градиент вычисляется с помощью `autograd`. Затем реализуется функция одномерного поиска оптимального шага и сам метод наискорейшего спуска:

```python
import numpy as np
from scipy.optimize import minimize_scalar
from autograd import grad
import autograd.numpy as anp

def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

grad_rosenbrock = grad(rosenbrock)

def line_search(f, x, direction):
    obj_func = lambda alpha: f(x - alpha * direction)
    result = minimize_scalar(obj_func)
    return result.x

def gradient_descent_with_logging(f, grad_f, x0, epsilon=1e-6, max_iter=20000):
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    f_values = [f(x)]
    
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < epsilon:
            print(f"Convergence reached at iteration {i}")
            break
        
        alpha = line_search(f, x, grad)
        x -= alpha * grad
        trajectory.append(x.copy())
        f_values.append(f(x))
    
    return x, f(x), trajectory, f_values
```

### 3. Запуск метода и вывод результатов

Запускается метод наискорейшего спуска с начальной точкой \((1.25, -0.45)\). Выводятся координаты точек на первых и последних итерациях, а также итоговая точка минимума:

```python
selected_starting_point = (1.25, -0.45)

final_x, final_f_val, trajectory, f_values = gradient_descent_with_logging(rosenbrock, grad_rosenbrock, selected_starting_point, 1e-6, 20000)

print("First 5 iterations:")
for i, point in enumerate(trajectory[:5]):
    print(f"Iteration {i}: Point = {point}")

print("\nLast 5 iterations:")
for i, point in enumerate(trajectory[-5:], start=len(trajectory) - 5):
    print(f"Iteration {i}: Point = {point}")

print(f"\nMinimum point: {final_x}")
```

### 4. Сравнение различных методов оптимизации

Производится сравнение различных методов оптимизации, включая NAG, Heavy-ball и Adam. Выводы делаются на основе количества итераций, необходимых для сходимости:

```markdown
Из полученных результатов видно, что алгоритмы наискорейшего спуска, NAG, Heavy-ball и Adam сошлись к минимуму функции Розенброка.

1. **Наискорейший спуск:**  Алгоритм наискорейшего спуска сходится к минимуму за 601 итерацию.
2. **Nesterov Accelerated Gradient (NAG):** Метод NAG сходится к минимуму за 3020 итераций.
3. **Heavy-ball:** Метод Heavy-ball сходится к минимуму за 3024 итерации.
4. **Adam:** Метод Adam сходится к минимуму за 15153 итерации.

Из полученных результатов можно сделать следующие выводы:
- Все рассмотренные методы сошлись к минимуму функции Розенброка.
- Методы NAG и Heavy-ball сходятся к минимуму более быстро по сравнению с наискорейшим спуском.
- Метод Adam требует большего количества итераций для сходимости по сравнению с другими методами.
```

Графическое представление траекторий движения к экстремуму также подтверждает сходимость всех алгоритмов к глобальному минимуму функции Розенброка.

Таким образом, данная работа иллюстрирует как аналитические, так и численные методы оптимизации, показывая их применимость и эффективность на примере функции Розенброка.
