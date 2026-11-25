/* Sizes are from Mference/README.md (short, blurb, maker and install are site copy).
   disk = install size in GB. need = the memory the app's MemoryFit estimates for
   the model (install x resident share + headroom); like the app, the picker says a
   model runs when need is at most 90% of the Mac's memory and the Mac clears the
   model's floor (minRam: Deep and Bonsai want more than 16 GB, Inkling 2 Mini 32 GB). */
window.MF = (function(){
  var HOSTS=[
    {id:"8",ram:8,label:"8 GB",mac:"M2 Air"},
    {id:"16",ram:16,label:"16 GB",mac:"M4"},
    {id:"24",ram:24,label:"24 GB",mac:"M5"},
    {id:"32",ram:32,label:"32 GB",mac:"M1 Max"},
    {id:"256",ram:256,label:"256 GB",mac:"M3 Ultra"}
  ];
  var FAMS=[
    {id:"quick",short:"Quick",blurb:"Everyday chat, mail and scanning",maker:"Qwen",install:"app",name:"Quick (Qwen3.5-4B)",p:4,disk:2.9,need:4.4},
    {id:"deep",short:"Deep",blurb:"Longer reasoning and writing",maker:"Qwen",install:"app",name:"Deep (Qwen3.8-27B)",p:27,disk:12.1,need:14.6,minRam:17},
    {id:"bonsai",short:"Bonsai",blurb:"A ternary 27B, lighter than Deep",maker:"PrismML",install:"app",name:"Bonsai 2.5 27B",p:27,disk:7.9,need:10.4,minRam:17},
    {id:"qwen36",short:"Qwen 3.6",blurb:"A 35B model for harder questions",maker:"Qwen",install:"terminal",name:"Qwen 3.6",p:35,disk:19.6,need:7.9},
    {id:"inkling",short:"Inkling 2 Mini",blurb:"For Macs with 32 GB and up",maker:"Thinking Machines",install:"terminal",name:"Inkling 2 Mini",p:26,disk:17.7,need:14.4,minRam:32},
    {id:"maple",short:"Maple",blurb:"Capable, and light on memory",maker:"deepgrove",install:"terminal",name:"Maple Preview",p:20,disk:6.6,need:8.1},
    {id:"minicpm",short:"MiniCPM5",blurb:"Small and fast",maker:"OpenBMB",install:"terminal",name:"MiniCPM5-2B",p:2,disk:1.43,need:2.4},
    {id:"minicpm1b",short:"MiniCPM5 1B",blurb:"The smallest, for any Mac",maker:"OpenBMB",install:"terminal",name:"MiniCPM5-1B",p:1,disk:0.6,need:1.35}
  ];
  function status(f,h){
    if((f.minRam&&h.ram<f.minRam)||f.need>h.ram*0.9) return "need";
    return "fits";
  }
  function fmtGB(x){return x>=10?String(Math.round(x)):(x>=1?String(+x.toFixed(2)).replace(/\.?0+$/,""):String(+x.toFixed(2)));}
  function pickHost(){var id="";try{id=location.hash.replace("#","");}catch(e){}return HOSTS.filter(function(h){return h.id===id;})[0]||HOSTS[2];}
  function setHash(id){try{history.replaceState(null,"","#"+id);}catch(e){}}
  return {HOSTS:HOSTS,FAMS:FAMS,status:status,fmtGB:fmtGB,pickHost:pickHost,setHash:setHash};
})();
