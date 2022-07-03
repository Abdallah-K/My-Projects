

$(function(){
  $(".btn").click(()=>{
    var search = $(".search").val();
    if (search ==""){
      alert("Please enter episde")
    }else{
      $.ajax({
        type:"GET",
        datatype:"json",
        url:`http://127.0.0.1:8000/${search}`,
        success:function(data){
          $(".title h1").html(data.Title);
          $(".sage h1").html(data.Sage);
          $(".type h1").html(data.Type);
          $(".date h1").html(data.AirDate);
          $(".votes h1").html(data.Votes);
          $(".rates h1").html(data.Rate);
        }
      })
    }
  })
})