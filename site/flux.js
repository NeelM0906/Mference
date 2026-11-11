/* Experts grid: every cell is one expert of one layer (Qwen3.8-Flash-Next: 512 experts, 10 called per token).
   sparse = only the called experts are lit (in memory); full = every cell lit (the whole model loaded). */
window.MFlux=(function(){
  function mount(cv,o){
    var cols=o.cols,rows=o.rows,N=cols*rows,TOP=o.top||10,gap=o.gap||3;
    var on=o.on,off=o.off,mode=o.mode||"sparse";
    var reduce=false; /* owner override: the animation always plays, even with Reduce Motion on */
    var w=cv.clientWidth||o.width||600,dpr=Math.min(window.devicePixelRatio||1,2);
    var s=Math.max(3,Math.floor((w-gap*(cols-1))/cols)),W=cols*s+(cols-1)*gap,H=rows*s+(rows-1)*gap;
    cv.style.width=W+"px";cv.style.height=H+"px";cv.width=Math.round(W*dpr);cv.height=Math.round(H*dpr);
    var ctx=cv.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);
    if(!ctx.roundRect)ctx.roundRect=function(x,y,w,h){this.rect(x,y,w,h);};
    var a=new Array(N).fill(0),t=new Array(N).fill(0),pu=new Array(N).fill(0),at=new Array(N).fill(0),r=Math.max(1,s*.22);
    function pick(now){
      var old=t.slice(),n=0,g=0;t.fill(0);
      while(n<TOP&&g++<2000){var i=(Math.random()*N)|0;if(!t[i]){t[i]=1;n++;at[i]=(old[i]||reduce)?now:now+Math.random()*520;}}
    }
    if(mode==="full"){t.fill(1);a.fill(1);} else pick(0);
    function cell(x,y,k,p){
      var z=s*(1+.55*p),o=(z-s)/2;
      if(o>0.2){ctx.globalAlpha=.22*p;ctx.fillStyle=on;ctx.beginPath();ctx.roundRect(x-o*2.2,y-o*2.2,s+o*4.4,s+o*4.4,r*2);ctx.fill();}
      ctx.globalAlpha=k;ctx.fillStyle=on;ctx.beginPath();ctx.roundRect(x-o,y-o,z,z,r);ctx.fill();ctx.globalAlpha=1;
    }
    function draw(){
      ctx.clearRect(0,0,W,H);
      var lit=[];
      for(var i=0;i<N;i++){
        var x=(i%cols)*(s+gap),y=((i/cols)|0)*(s+gap);
        ctx.fillStyle=off;ctx.beginPath();ctx.roundRect(x,y,s,s,r);ctx.fill();
        if(a[i]>.01)lit.push(i);
      }
      for(var j=0;j<lit.length;j++){var q=lit[j];cell((q%cols)*(s+gap),((q/cols)|0)*(s+gap),a[q],pu[q]);}
    }
    if(mode==="full"){a=t.slice();draw();if(o.onTick)o.onTick(TOP,N);return;}
    var last=0,next=0,run=true;
    function frame(now){
      if(!run)return;var dt=Math.min(.05,(now-last)/1000||0);last=now;
      if(now>=next){pick(now);next=now+(reduce?2600:(o.every||1500));if(o.onTick)o.onTick(TOP,N);}
      for(var i=0;i<N;i++){
        if(t[i]){ if(now>=at[i]){ if(!reduce&&a[i]<.5&&pu[i]<.5)pu[i]=1; a[i]+=(1-a[i])*Math.min(1,dt*(reduce?2.2:10));} }
        else a[i]+=(0-a[i])*Math.min(1,dt*(reduce?2.2:3.2));
        pu[i]*=Math.exp(-dt*5);
      }
      draw();requestAnimationFrame(frame);
    }
    document.addEventListener("visibilitychange",function(){if(document.hidden)run=false;else if(!run){run=true;last=performance.now();requestAnimationFrame(frame);}});
    if(o.onTick)o.onTick(TOP,N);
    requestAnimationFrame(frame);
  }
  return {mount:mount};
})();
