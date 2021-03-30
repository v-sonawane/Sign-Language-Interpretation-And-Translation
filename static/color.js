var path = window.location.pathname;
var page = path.split("/").pop();
console.log(page);
if (page=="index.html"){
    var element=document.getElementById("index");
    element.classList.add('ongetName');
    console.log(element.classList);
}
else{
    var element=document.getElementById("instructions");
    console.log(element.classList);
    element.classList.add('ongetName');
    console.log(element.classList);
}