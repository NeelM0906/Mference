/* Every value is from Mference/README.md. gb = memory the family reports; disk = install size; sp = measured decode tok/s by host. */
window.MF = (function(){
  var HOSTS=[
    {id:"8",ram:8,label:"8 GB",mac:"M2 Air"},
    {id:"16",ram:16,label:"16 GB",mac:"M4"},
    {id:"24",ram:24,label:"24 GB",mac:"M5"},
    {id:"256",ram:256,label:"256 GB",mac:"M3 Ultra"}
  ];
  var FAMS=[
    {id:"dsv4",name:"DeepSeek-V4-Flash",p:284,disk:91,mem:{d:5.7},sp:{"256":[5.6,6.7]},note:"experimental"},
    {id:"inkling",name:"Inkling-Small",p:276,disk:148,mem:{d:9},sp:{"24":[3.0,3.7],"256":[5.3,7.1]}},
    {id:"flash",name:"Qwen3.8-Flash-Next",p:180,disk:175,mem:{d:2.36,"256":68},sp:{"256":[18.2,21.0]},note:"16 slots, 2.36 GB; 256 GB hosts map a 68 GiB pool"},
    {id:"qwen36",name:"Qwen 3.6",p:35,disk:19.6,mem:{d:1.45,"256":6.8},sp:{"24":[23.5,29.3],"256":[36.1,42.2]}},
    {id:"qwen38",name:"Qwen 3.8",p:27,disk:15,mem:{d:15},minRam:24,sp:{"24":[15.0,15.0],"256":[38.4,39.4]}},
    {id:"gemma4",name:"Gemma 4",p:26,disk:14.3,mem:{d:2},sp:{"8":[5.1,6.3],"24":[31,35],"256":[17.1,18.7]}},
    {id:"qat",name:"Gemma 4 QAT",p:26,disk:14.45,mem:{d:2.16},sp:{"16":[7.11,7.11]}},
    {id:"maple",name:"Maple Preview",p:20,disk:6.6,mem:{d:0.49,"16":1.21},sp:{"16":[18.9,24.6],"256":[38.5,38.5]}},
    {id:"minicpm",name:"MiniCPM5",p:2,disk:1.43,mem:{d:0.123},sp:{"24":[65.5,65.8]}},
    {id:"glm",name:"GLM-5.3-Flash",p:320,disk:181,mem:{d:181},sp:{}}
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
