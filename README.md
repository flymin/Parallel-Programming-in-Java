# Parallel Programming in Java

## Task Parallel

Task Parallel 包括 task 创建和 task 终止两部分主要操作。

- 使用 *async* 指明那些 task 是可以被并行执行的
- 使用 *finish block* 等待任务同步结束，回到顺序执行的部分

使用 [PCDP](https://github.com/habanero-rice/pcdp) 包可以很好地在代码中支持 *async* 和 *finish block* 标注

### Fork-Join Framework

Java 标准中需要使用 *Fork-Join Framework* 。将可以并行的部分使用 *fork* 创建一个新的进程执行，在需要同步的地方使用 *join* 等待该进行的结束。Java 中的封装机制使得 *fork* 和 *join* 函数对应一个类实例执行，即：

```Java
instance.fork();
instance.function();
instance.join();
```

完成一个并行 task 的部署和同步。

### 计算图的使用

进行程序设计时，可以使用计算图——有向无环图——来表示 task 级并行的执行顺序与相互关系。

计算图的组成：

- 节点：表示并行模型中的 task
- 有向边：task 之间的相互转移关系
  - Continue edge：表示需要顺序执行的 task
  - Fork edge：表示 fork 操作对应的 child task
  - Join edge：表示 join 操作，将不同的 child task 同步

计算图的性质：

- 如何判断两个任务之间能否并行？

  图中两个节点之间如果有直接路径则表示有相互关系，不可并行；如果两个节点之间没有路径表示可以并行

- 数据竞争（data race）

  要防止相互并行的节点之间有读写访问的竞争性

- 衡量并行模型的表现

  - WORK：表示模型中 task 串行时的表现，即图中所有节点表示的 task 的执行时间和
  - SPAN：图中最长的路径长度，表示整个并行模型的执行时间
  - ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bideal%20parallelism%7D%3D%5Cfrac%7BWORK%28G%29%7D%7BSPAN%28G%29%7D)

### 多处理器规划与并行加速

![](http://latex.codecogs.com/gif.latex?T_{p}) 表示有 ![](http://latex.codecogs.com/gif.latex?p) 个处理器的执行时间。多处理器的执行时间不仅和任务本身的执行时间、计算图有关，而且和任务规划中的执行方式有关。满足：

![](http://latex.codecogs.com/gif.latex?T_%7B1%7D%3DWORK%5C%5CT_%7B%5Cinfin%7D%3DSPAN%5C%5CT_%7B%5Cinfin%7D%5Cle%7BT_%7Bp%7D%7D%5Cle%7BT_%7B1%7D%7D)

由此得到规划方式的加速比计算：

![](http://latex.codecogs.com/gif.latex?SPEEDUP%3D%5Cfrac%7BT_%7B1%7D%7D%7BT_%7Bp%7D%7D%20%5C%5C%20SPEEDUP%5Cle%7Bp%7D%20%5C%5C%20SPEEDUP%5Cle%7B%5Ctext%7Bideal%20parallelism%7D%7D)

### 阿姆达尔定律（Amdahl‘s law）

给定一组 task，需要多少处理器能够得到最佳的加速比？

- 已知计算图的情况下，可以通过计算 ![](http://latex.codecogs.com/gif.latex?\frac{WORK}{SPAN}) 得到 ![](http://latex.codecogs.com/gif.latex?SPEEDUP) 的上界

- 未知计算图的情况下，根据 Amdahl's law，考虑 task 中顺序执行的部分所占的比例 ![](http://latex.codecogs.com/gif.latex?q)，![](http://latex.codecogs.com/gif.latex?SPEEDUP\le{\frac{1}{q}})

  > ![](http://latex.codecogs.com/gif.latex?%5Cbecause%7BSPAN%5Cge%7Bq%5Ctimes%7BWORK%7D%7D%7D%20%5C%5C%20%5Ctherefore%7BSPEEDUP%5Cle%7B%5Cfrac%7BWORK%7D%7BSPAN%7D%7D%3D%5Cfrac%7BWORK%7D%7Bq%5Ctimes%7BWORK%7D%7D%3D%5Cfrac%7B1%7D%7Bq%7D%7D)

### 程序实例

#### 使用 async-finish API

```Java
package edu.coursera.parallel;

import static edu.rice.pcdp.PCDP.async;
import static edu.rice.pcdp.PCDP.finish;

public final class Compare {
    private static double sum1;
    private static double sum2;

    Compare() {

    }

    public static void main(String[] args) {
        double[] array = new double[20000000];
        for (int i = 0; i < 20000000; i++) {
            array[i] = i + 1;
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(String.format("This is run for %d/5", i + 1));
            parallel(array);
            sequential(array);
        }
    }

    public static void sequential(final double... array) {
        sum1 = 0.;
        sum2 = 0.;
        long start = System.nanoTime();
        for (int i = 0; i < array.length / 2; i++) {
            sum1 += 1 / array[i];
        }

        for (int j = array.length / 2; j < array.length; j++) {
            sum2 += 1 / array[j];
        }
        double seqResult = sum1 + sum2;
        long seqTime = System.nanoTime() - start;
        System.out.println(String
            .format("sequential version time: %#.2f, result is %#.4f",
                seqTime / 1e6, seqResult));
    }

    public static void parallel(final double... array) {
        sum1 = 0.;
        sum2 = 0.;
        long start = System.nanoTime();
        finish(() -> {
            async(() -> {
                for (int i = 0; i < array.length / 2; i++) {
                    sum1 += 1 / array[i];
                }
            });
            for (int j = array.length / 2; j < array.length; j++) {
                sum2 += 1 / array[j];
            }
        });

        double parResult = sum1 + sum2;
        long parTime = System.nanoTime() - start;
        System.out.println(String
                .format("parallel version time: %#.2f, result is %#.4f",
                    parTime / 1e6, parResult));
    }
}
```

Result:

```
This is run for 9/5
parallel version time: 11.39, result is 17.3885
sequential version time: 22.93, result is 17.3885
This is run for 10/5
parallel version time: 11.87, result is 17.3885
sequential version time: 21.44, result is 17.3885
```



#### 使用 fork-join Framework

[example](https://github.com/flymin/Parallel-Programming-in-Java/blob/master/miniproject_1/src/main/java/edu/coursera/parallel/ReciprocalArraySum.java#L135)

## functional parallelism

Functional parallelism 理解的重点在于 Future Tasks 和 Future objects（或称 promise objects）。

- Future tasks：一个包含返回值的计算任务，其他任务可以通过调用来要求计算或者直接访问（当这个返回值已经计算时）这个返回值
- Future objects：指为future tasks提供访问方式的对象

future task可以理解为是一种描述计算图的方式，通过 future 表示当前任务计算完成之后的结果，供后面的步骤调用，因此在建模时就能够很自然的描述出全部计算图的依赖关系，在依据函数相互的调用关系就能完成并行建模。在 future 模型中有两个关键的问题：

- Assignment：即对于一个 future task，他接受一个输入并产生唯一的输出，两者在计算开始之后都不能修改
- Blocking read：为了使得 future 模型体现依赖关系，在前一个步骤没有进行完毕的时候要对当前模型进行阻塞，等待前序步骤完成之后再继续执行，这能够避免数据竞争的问题。

![例1](https://img-blog.csdnimg.cn/20190707220508308.png)

上图是来自课程Quiz的一道题目，更好地展示 future 的分析作用，其中可以看出

- S2 对 S1 没有依赖关系
- S3 需要在 S1 执行完后执行
- S4 不依赖 B 中的任务，但要等到 S1 执行完毕
- S2 需要等到 S1-S3 都执行完毕才能执行
- future 内部要保持顺序执行关系

通过以上分析，我认为从 future 块到计算图的还原最好采用逆向分析的方式，即应用递归程序的思想（实际上框架似乎也是这么分析的）

### 使用 Fork/Join 框架进行编程

框架的使用方法和之前的方式较为类似，关键点还是在于实现 `compute()` 函数进行计算，并使用 `join()` 函数完成阻塞操作，主要有以下需要特别注意的点：

- future task 需要继承 RecursiveTask 类而不是 RecursiveAction 类
- 使用方法与之前类似，但是  `compute()` 函数是有返回值的，不能是 `void` 类型
- join 会发生阻塞并等待同步，同时会提供返回值

### Memoization

相当于给计算结果建立 Cache，例如：对于 ![](http://latex.codecogs.com/gif.latex?y_{1}=G(x_{1})) ，当计算完毕之后，不仅仅赋值给 ![](http://latex.codecogs.com/gif.latex?y_{1}) 。会同时记录下这个结果来自于 ![](http://latex.codecogs.com/gif.latex?future\{G,x\}) ，因此在下次调用这个结果时，就可以通过直接查表获取到结果从而避免计算。

Memoization 是动态规划算法的设计来源，即通过使用存储来换取运算时间上的优化。

因为依旧是使用 future 模型进行建模，因此这里还是要求实现一个 `get()` 操作来获取计算出来的结果的值。

### Java Streams

这是 Java 8 中加入的新特性，主要针对一个 for 循环，可以通过调用 parallel stream 实现并行化得循环计算。

```java
students.stream().forEach(s \rightarrow→ System.out.println(s));
students.stream()
    .filter(s -> s.getStatus() == Student.ACTIVE)
    .mapToInt(a -> a.getAge())
    .average();
```

计算得关键点有两个，即 `filter` 用来过滤集合中符合条件得元素，`map` 用来调用集合中每个元素的计算值。使用 stream 的的方式就可以方便的建立并行化的计算了

```java
tudents.parallelStream()
// or
Stream.of(students).parallel()
```

### Determinism

**functional determinism**：指函数在相同的输入下会有相同输出的性质

**structural determinism**：指程序中对于相同的输入会产生相同计算图的性质

程序中的不确定的性通常是由于数据竞争导致的

**data race freedom = functional determinism + structural determinism**

- 有数据竞争出现的程序并不一定是非确定的程序
- 没有数据竞争也不一定能保证确定性
- 使用课程中介绍的模型，在不发生数据竞争的前提下就可以保证是确定性程序

**benign non-determinism**：指程序中虽然不能保证确定性，但是非确定的结果对于程序的正确性来说是可以接受的

### 使用 Stream 实例

#### [simple example](https://github.com/flymin/Parallel-Programming-in-Java/blob/master/miniproject_2/src/main/java/edu/coursera/parallel/StudentAnalytics.java#L151)

#### [some more complex](https://github.com/flymin/Parallel-Programming-in-Java/blob/master/miniproject_2/src/main/java/edu/coursera/parallel/StudentAnalytics.java#L109)


相比之下，这个例子更能体现出 stream 的易用性，在这里，collect 被用作一个收集器进行**分类汇总**，然后将结果传递给下游收集器 `Collectors.counting()` 进行进一步的 **reduce** 计算。

出了上面给出的例子，`reduce` 也是一个功能强大的 API，更多信息参考：

[Java8-15-Stream 收集器 01-归约与汇总+分组](https://houbb.github.io/2019/02/27/java8-15-stream-collect-01#%E7%BB%9F%E4%B8%80%E8%8E%B7%E5%8F%96%E6%B1%87%E6%80%BB%E4%BF%A1%E6%81%AF-summarizingxxx)

[Java Streams，第 2 部分- 使用流执行聚合-轻松地分解数据](https://www.ibm.com/developerworks/cn/java/j-java-streams-2-brian-goetz/index.html)



## Parallel Loops

循环结构在编程实践中是一类很常见的构建模型。程序中的循环结构主要可分为：

- pointer-chasing loop

  ```java
  for(p = HEAD; p != NULL; p.NEXT) {
  	compute(p);
  }
  ```

  这类循环结构中，因为每次循环处理的指针对象都是相互独立的，因此任务完全可以分开进行处理。可以简单地采用将循环看做 async task 使得计算模型并行化

  ```java
  FINISH{
  	for(p = HEAD; p != NULL: p = p.NEXT) {
  		ASYNC
  		compute(p);
  	}
  }
  ```

- Iteration loop

  ```java
  for (i : [0 : N-1]) {
  	A[i] = B[i] + C[i];
  }
  ```

  这类模型和pointer-chasing loop最大的不同处在于这类循环中，循环的次数是可以提前知道的。

  - 简单的使用`forall`代替`for`来调用现有的API就可以自动的并行化。

  - 或者调用 stream 来并行化 for 循环。

    ```java
    a = IntStream.rangeClosed(0, N-1).parallel().toArray(i -> b[i] + c[i]);
    ```

    但具有多个返回值的时候还是使用`forall`构建会更清晰简单。

### 矩阵乘法的并行化

一个串行化得矩阵乘法实例

```java
for([i, j] : [0:N-1, 0:N-1]) {
	c[i][i] = 0;
	for(k : [0:N-1]) {
		c[i][j] += A[i][k] * B[k][j]
	}
}
```

并行化时，将外层循环的 `for` 替换为使用 `forall` 。内层的 k 循环是不能进行进一步的并行化的，因为在计算的过程中如果进行并行化就会产生数据竞争。

### 循环并行的屏障（同步）

在一个并行循环的模型中，循环体部分可以加入 barrier。作用是进行进程之间的同步，进行第一次执行到 barrier 会等待，待所有线程都到达之后再一次开始。通过加入 barriers，for的循环体被分割为不同的 phases 进行操作，进程会在 phase 之间进行同步，然后继续并行执行。

### 注意事项

使用 `forall` 循环时，将全部循环都创建 task 有时并不是好的方案，应当根据具体的硬件环境（即处理器核心等）创建适合的并行模式，在尽量完全的利用硬件运算优势的前提下减少因为 task 分配造成的开销。

例如对于计算向量和的程序并行化，将整个向量分块会是一个较好的方法，常用的分块方法有两种：

- 使用固定大小的block进行分块
- 使用 cyclic 模式进行分块，即 ![](http://latex.codecogs.com/gif.latex?i%5Ctext%7B%20mod%20%7D%20NG) 的结果作为分块依据（这通常适用于向量任务中计算量分配不均衡的时候，使用这种方法能够最大程度上地平衡 workload）

## Dataflow Synchronization and Pipelining

### Split-phase Barriers with Java Phasers

在一般的使用 barrier 的任务中，通常同步操作本身是需要消耗开销的。实际上，这部分开销是可以整合在程序的其他部分中的，即将 barrier 放置在程序的某个步骤中，在执行的同时进行同步。

在 barrier 执行时，实际上分为几个步骤，即 ARRIVE-AWAIT-ADVANCE。因此，显然将这些步骤分离，将不是必须等待的步骤提前执行，让程序拥有更多的“等待缓冲”时间是一个较好的优化思路。phaser object 可以分为两部分：ARRIVE 和 AWAIT-ADVANCE。通常的 barrier 模型可以表示为

```java
forall(i : [1:N]) {
	print("HELLO");
  myid = LOOKUP(i);
  NEXT;
  print("bye" + myid);
}
```

使用 phaser 之后，模型可以表示为：

```java
// initialize phaser ph	for use by n tasks ("parties") 
Phaser ph = new Phaser(n);
// Create forall loop with n iterations that operate on ph 
forall (i : [0:n-1]) {
  print HELLO, i;
  int phase = ph.arrive();
  
  myId = lookup(i); // convert int to a string

  ph.awaitAdvance(phase);
  print BYE, myId;
}
```

ARRIVE 表示当前线程进入了这个 phaser ，但并不需要等待，仍然可以继续执行一些本地化的操作。执行到 AWAIT ADVANCE 时才是真正需要等待同步的地方。因此，使用 phaser 相当于将同步操作和本身需要执行的步骤进行了一定程度的重叠，使关键路径时间缩短。

### Point-to-Point Synchronization with Phasers

使用 phase 的同步模型可以使得当数据依赖关系复杂时，保证在效率最高的情况下没有数据竞争地完成任务。以下面的计算依赖关系为例：

| **Task 0**                | **Task 1**                | **Task 2**                |
| ------------------------- | ------------------------- | ------------------------- |
| 1a:X=A();//cost=1         | 1b:Y=B();//cost=2         | 1c:Z=C();//cost=3         |
| *2a:ph0.arrive();*        | *2b:ph1.arrive();*        | *2c:ph2.arrive();*        |
| *3a:ph1.awaitAdvance(0);* | *3b:ph0.awaitAdvance(0);* | *3c:ph1.awaitAdvance(0);* |
| 4a:D(X,Y);//cost=3        | *4b:ph2.awaitAdvance(0);* | 4c:F(Y,Z);//cost=1        |
|                           | 5b:E(X,Y,Z);//cost=2      |                           |

上面的任务中，如果不使用 phase 模型，就必须等所有 task 的第一步完成之后进行一次同步，再共同开启后面的程序。但使用 phase 模型可以让任务更加**精确**地知道自己在等待的任务，一旦该任务完成便可以立即开始。

### Pipeline Parallelism

即流水线并行，适用于可分为多个独立步骤并且需要处理序列化的多个独立输入的任务。

在实现时，每个步骤之间只需要等待前一个步骤完成就可以执行。使用流水线模型，假设要处理n个输入，每次处理需要p步，时间都是1，则![](http://latex.codecogs.com/gif.latex?WORK=n\times{p},CPL=n+p−1,PAR=\frac{WORK}{CPL}=\frac{np}{n+p−1}) ，当 n 远大于 p 时，效率趋近于 p，这已经达到了最理想的情况了。

### Data Flow Parallelism

数据流并行模型意在使用 async 来实现并行计算图的构建，从而更加具体地体现并使用数据操作之间的依赖关系。假设现有的数据依赖为：*A → C, A → D, B → D, B → E*，则可以构建以下模型：

```java
async( () -> {/* Task A */; A.put(); } ); // Complete task and trigger event A
async( () -> {/* Task B */; B.put(); } ); // Complete task and trigger event B
asyncAwait(A, () -> {/* Task C */} );	    // Only execute task after event A is triggered 
asyncAwait(A, B, () -> {/* Task D */} );	  // Only execute task after events A, B are triggered 
asyncAwait(B, () -> {/* Task E */} );	    // Only execute task after event B is triggered
```



