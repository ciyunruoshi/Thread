1.```Thread```接口是线程基本的类
```
Thread thread = new MyThread()；
thread.start();
```
这就是继承了```Thread```接口的创建方式，可以看到如果只使用Thread，那么使用，只能在继承的时候（模板模式）实现具体的```run()```方法，
一旦创建就不能更改线程的功能，也就是不能共享代码。

2.在这种情况下，```Runnable```接口就解决了问题，```Runnable```类作为“任务”传入```Thread```类，```Thread调用“任务”方法run()来完成任务```，
这样就带来了一个好处，那就是不同线程```Thread()```可以执行相同任务，也就是可以共享代码。

3.FutureTask(class implements RunnableFuture<T> extends Future<T>) 和 Callable（interface）组合构建可以获得返回值的线程
  
4.ThreadPool:优点，将任务提交和执行解耦，线程池全部是基于RunnableTask和Callable接口来实现的，提交的Runnable也全部是转换成Callable类
实现，想想也是线程池是后面版本才出来，而且Future和Callable是功能最为强大的线程实现方式，当然要使用最强大的，保证这个框架能够应付各种情况。

5.线程的转态：新建（new） 运行（Running and Runnable） 等待（wait） 超时等待（Timed_wait） 终止（stop）

6.```Executors```提供和很多构建不同线程池的方法，但是不允许使用此类创建线程池,为什么？
```
public class Executors {
   public static ExecutorService newFixedThreadPool(int nThreads) {
        return new ThreadPoolExecutor(nThreads, nThreads,
                                      0L, TimeUnit.MILLISECONDS,
                                      new LinkedBlockingQueue<Runnable>());
    }
    
    
    public static ExecutorService newSingleThreadExecutor() {
        return new FinalizableDelegatedExecutorService
            (new ThreadPoolExecutor(1, 1,
                                    0L, TimeUnit.MILLISECONDS,
                                    new LinkedBlockingQueue<Runnable>()));
    }
    
    public static ExecutorService newCachedThreadPool() {
        return new ThreadPoolExecutor(0, Integer.MAX_VALUE,
                                      60L, TimeUnit.SECONDS,
                                      new SynchronousQueue<Runnable>());
    }
}
```
可以看到```Executors```提供的方法都是由```ThreadPoolExecutor```类实现，而从参数上来看```new LinkedBlockingQueue<Runnable>()```提供的是
链表实现的阻塞队列，而默认的链表阻塞队列的容量是无限的，那么这会导致什么问题？

可以一直```submit()```Runnable实例，会耗尽内存并且溢出。
