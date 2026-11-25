/* Every value is from Mference/README.md (short, blurb, maker and install are site copy). gb = memory the family reports; disk = install size; sp = measured decode tok/s by host. */
window.MF = (function(){
  var HOSTS=[
    {id:"8",ram:8,label:"8 GB",mac:"M2 Air"},
    {id:"16",ram:16,label:"16 GB",mac:"M4"},
    {id:"24",ram:24,label:"24 GB",mac:"M5"},
    {id:"32",ram:32,label:"32 GB",mac:"M1 Max"},
    {id:"256",ram:256,label:"256 GB",mac:"M3 Ultra"}
  ];
  var FAMS=[
    {id:"quick",short:"Quick",blurb:"Everyday chat, mail and scanning",maker:"Qwen",install:"app",name:"Quick (Qwen3.5-4B)",p:4,disk:2.9,mem:{d:1.3},sp:{"32":[50,50]}},
    /* Deep: 3.1 GB plus 12.1 GB of mapped weights; the app offers it above 16 GB. */
    {id:"deep",short:"Deep",blurb:"Longer reasoning and writing",maker:"Qwen",install:"app",name:"Deep (Qwen3.8-27B)",p:27,disk:12.1,mem:{d:15.2},minRam:17,sp:{"32":[17,17]}},
    {id:"qwen36",short:"Qwen 3.6",blurb:"A 35B model for harder questions",maker:"Qwen",install:"terminal",name:"Qwen 3.6",p:35,disk:19.6,mem:{d:1.45,"256":6.8},sp:{"24":[23.5,29.3],"32":[29,29],"256":[36.1,42.2]}},
    {id:"inkling",short:"Inkling 2 Mini",blurb:"For Macs with 32 GB and up",maker:"Thinking Machines",install:"terminal",name:"Inkling 2 Mini",p:26,disk:17.7,mem:{d:14.4},minRam:32,sp:{"32":[26,26]}},
    {id:"maple",short:"Maple",blurb:"Capable, and light on memory",maker:"deepgrove",install:"terminal",name:"Maple Preview",p:20,disk:6.6,mem:{d:0.49,"16":1.21},sp:{"16":[18.9,24.6],"32":[33,33],"256":[38.5,38.5]}},
    {id:"minicpm",short:"MiniCPM5",blurb:"The smallest and fastest",maker:"OpenBMB",install:"terminal",name:"MiniCPM5",p:2,disk:1.43,mem:{d:0.123},sp:{"24":[65.5,65.8],"32":[72,72]}}
  ];
  function memFor(f,h){return f.mem[h.id]!=null?f.mem[h.id]:f.mem.d;}
  function status(f,h){
    var m=memFor(f,h);
    if(m>h.ram||(f.minRam&&h.ram<f.minRam)) return "need";
    return f.sp[h.id]?"measured":"fits";
  }
  function fmtGB(x){return x>=10?String(Math.round(x)):(x>=1?String(+x.toFixed(2)).replace(/\.?0+$/,""):String(+x.toFixed(2)));}
  function fmtSp(a){return a[0]===a[1]?String(a[0]):a[0]+"–"+a[1];}
  function pickHost(){var id="";try{id=location.hash.replace("#","");}catch(e){}return HOSTS.filter(function(h){return h.id===id;})[0]||HOSTS[2];}
  function setHash(id){try{history.replaceState(null,"","#"+id);}catch(e){}}
  return {HOSTS:HOSTS,FAMS:FAMS,memFor:memFor,status:status,fmtGB:fmtGB,fmtSp:fmtSp,pickHost:pickHost,setHash:setHash};
})();
