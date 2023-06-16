function preprocess(){
  var xhr=new XMLHttpRequest()
  document.getElementById('spinner').style.display='block'
  xhr.open('get','/preprocessing',true)
  xhr.onload = () => {
    if (xhr.readyState === xhr.DONE) {
      if (xhr.status === 200) {
        document.getElementById('spinner').style.display='none'
      }
    }
  };
  xhr.send()
}
var c=0
function segment(){
    document.getElementById('spinner').style.display='block'
    var xhr=new XMLHttpRequest()
    xhr.open('get','/segmentaion',true)
    xhr.onload = () => {
        if (xhr.readyState === xhr.DONE) {
          if (xhr.status === 200) {
            document.getElementById('spinner').style.display='none'
          }
        }
      };
    xhr.send()
}
function dwt()
{
    c++;
    if(c%2==1){
    document.getElementById("dwt").style.display = "none";
    }
   else{
        document.getElementById("dwt").style.display = "block";
    }
}
// Add your custom JavaScript code here

// Example: Display an alert when the page is loaded

// Example: Toggle a class on click

var navLinks = document.getElementsByClassName('nav-link');

    for (var i = 0; i < navLinks.length; i++) {
      navLinks[i].addEventListener('click', function() {
        this.classList.toggle('active');
      });
    }

function function1()
{
    var v=document.getElementById('datachoice').value
    if(v=='Uploaded new preprocessed Data')
    {
        document.getElementById('preprocessUpload').style.display='block'
    }
    else{
        document.getElementById('preprocessUpload').style.display='none'
    }
}


   