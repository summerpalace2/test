package com.example.next;


import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;

import com.google.gson.Gson;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;

public class JsonDatapost {
    private Handler mHandler1;
    private HashMap<String,String> params;

    public JsonDatapost(Handler handler, HashMap<String,String> params )
    {
        this.mHandler1 = handler;
        this.params=params;
    }
    public String StreamToString(InputStream in)
    {
        StringBuilder sb = new StringBuilder();
        String Line;
        BufferedReader reader = new BufferedReader(new InputStreamReader(in));
        try {
            while ((Line = reader.readLine()) != null) {
                sb.append(Line);
                sb.append('\n');

            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                in.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return sb.toString();
    }

    public void sendPostRequest(String mUrl1)
    {
        //lambda表达式，相当于其中new Runnable 并且重写⽅法
        new Thread(() -> {

            try {
                Looper.prepare();
                URL url = new URL(mUrl1);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");//设置请求⽅式为Post
                connection.setConnectTimeout(15000);//设置最⼤连接时间，单位为ms
                connection.setReadTimeout(15000);//设置最⼤的读取时间，单位为ms
                connection.setRequestProperty("Accept-Language", "zh-CN,zh;q=0.9");
                connection.setRequestProperty("Accept-Encoding", "gzip,deflate");
                StringBuilder dataToWrite = new StringBuilder();
                for (String key : params.keySet())
                {
                dataToWrite.append(key).append("=").append(params.get(key)).append("&");
               }
              connection.connect();
               OutputStream outputStream =
                     connection.getOutputStream();
                outputStream.write(dataToWrite.substring(0,
                      dataToWrite.length() - 1).getBytes());
               InputStream in = connection.getInputStream();//从接⼝处获取
                String responseData = StreamToString(in);//这⾥就是服务器返回的数据
                Message message = new Message();
                message.what = 100;
                message.obj = responseData;
                mHandler1.sendMessage(message);
                Log.d("lx", "sendGetNetRequest: " + responseData);
            } catch (Exception e) {
                e.printStackTrace();
            }
            Looper.loop();
        }).start();

    }

}
